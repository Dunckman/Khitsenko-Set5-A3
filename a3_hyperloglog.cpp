#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdio>

uint32_t murmur_hash3(const void *, int, uint32_t);

class RandomStreamGen {
private:
    size_t length_;
    std::vector<std::string> stream_;

public:
    RandomStreamGen() : length_(0) {
    }

    explicit RandomStreamGen(size_t length) : length_(length) { renew(); }

    void renew() {
        stream_.clear();
        stream_.reserve(length_);
        const std::string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-";
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist_chars(0, chars.size() - 1);
        std::uniform_int_distribution<size_t> dist_length(1, 30);
        for (size_t i = 0; i < length_; ++i) {
            size_t len = dist_length(gen);
            std::string str(len, ' ');
            for (size_t j = 0; j < len; ++j) {
                str[j] = chars[dist_chars(gen)];
            }
            stream_.push_back(std::move(str));
        }
    }

    [[nodiscard]] const std::vector<std::string> &get_stream() const { return stream_; }
    [[nodiscard]] size_t size() const { return length_; }
};

class HashFuncGen {
public:
    static uint32_t hash(const std::string &str) noexcept {
        return murmur_hash3(str.data(), static_cast<int>(str.size()), 42);
    }
};

class HyperLogLog {
private:
    size_t B_;
    size_t M_;
    std::vector<uint8_t> registers_;

public:
    explicit HyperLogLog(size_t B) : B_(B), M_(1u << B) {
        registers_.resize(M_, 0);
    }

    void reset() {
        std::fill(registers_.begin(), registers_.end(), 0);
    }

    void add(const std::string &str) {
        uint32_t h = HashFuncGen::hash(str);
        size_t idx = h >> (32 - B_);
        uint32_t remaining = (h << B_) | ((1u << B_) - 1);
        uint8_t rho = 1;
        while ((remaining & (1u << 31)) == 0 && rho < (32 - B_)) {
            rho++;
            remaining <<= 1;
        }
        if (rho > registers_[idx]) {
            registers_[idx] = rho;
        }
    }

    [[nodiscard]] double estimate() const {
        double z = 0.0;
        size_t v = 0;
        for (size_t i = 0; i < M_; ++i) {
            z += std::pow(2.0, -static_cast<double>(registers_[i]));
            if (registers_[i] == 0) v++;
        }
        double alpha;
        if (M_ == 16) alpha = 0.673;
        else if (M_ == 32) alpha = 0.697;
        else if (M_ == 64) alpha = 0.709;
        else alpha = 0.7213 / (1.0 + 1.079 / static_cast<double>(M_));

        double e = alpha * static_cast<double>(M_) * static_cast<double>(M_) / z;

        if (e <= 2.5 * static_cast<double>(M_) && v > 0) {
            e = static_cast<double>(M_) * std::log(static_cast<double>(M_) / static_cast<double>(v));
        }
        return e;
    }

    [[nodiscard]] size_t getB() const {
        return B_;
    }

    [[nodiscard]] size_t getM() const {
        return M_;
    }

    static size_t exact_count(const std::vector<std::string> &stream, size_t up_to) {
        std::unordered_set<std::string> unique(stream.begin(), stream.begin() + up_to);
        return unique.size();
    }
};

uint32_t murmur_hash3(const void *key, int len, uint32_t seed) {
    const uint8_t *data = (const uint8_t *) key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t *blocks = (const uint32_t *) (data);
    for (int i = 0; i < nblocks; ++i) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t *tail = (const uint8_t *) (data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
            [[fallthrough]];
        case 2: k1 ^= tail[1] << 8;
            [[fallthrough]];
        case 1: k1 ^= tail[0];
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            h1 ^= k1;
    }

    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    return h1;
}

void run_experiment(size_t B, size_t STREAM_SIZE, size_t NUM_STREAMS,
                    int STEP_PERCENT, const std::string &prefix) {
    std::vector<size_t> steps;
    for (int p = STEP_PERCENT; p <= 100; p += STEP_PERCENT) {
        steps.push_back(STREAM_SIZE * p / 100);
    }
    size_t num_steps = steps.size();

    std::vector<std::vector<double> > all_estimates(NUM_STREAMS, std::vector<double>(num_steps));
    std::vector<std::vector<size_t> > all_exact(NUM_STREAMS, std::vector<size_t>(num_steps));

    RandomStreamGen gen(STREAM_SIZE);

    for (size_t s = 0; s < NUM_STREAMS; ++s) {
        if (s > 0) gen.renew();
        const auto &stream = gen.get_stream();
        HyperLogLog hll(B);

        size_t step_idx = 0;
        for (size_t i = 0; i < STREAM_SIZE; ++i) {
            hll.add(stream[i]);
            if (step_idx < num_steps && (i + 1) == steps[step_idx]) {
                all_estimates[s][step_idx] = hll.estimate();
                all_exact[s][step_idx] = HyperLogLog::exact_count(stream, i + 1);
                step_idx++;
            }
        }
        std::cout << "B=" << B << " Stream " << (s + 1) << "/" << NUM_STREAMS << " done\n";
    }

    std::vector<double> mean_est(num_steps, 0.0);
    std::vector<double> mean_ex(num_steps, 0.0);
    std::vector<double> std_est(num_steps, 0.0);

    for (size_t t = 0; t < num_steps; ++t) {
        for (size_t s = 0; s < NUM_STREAMS; ++s) {
            mean_est[t] += all_estimates[s][t];
            mean_ex[t] += static_cast<double>(all_exact[s][t]);
        }
        mean_est[t] /= NUM_STREAMS;
        mean_ex[t] /= NUM_STREAMS;

        for (size_t s = 0; s < NUM_STREAMS; ++s) {
            double diff = all_estimates[s][t] - mean_est[t];
            std_est[t] += diff * diff;
        }
        std_est[t] = std::sqrt(std_est[t] / (NUM_STREAMS - 1));
    } {
        std::ofstream f("data/" + prefix + "_graph1.csv");
        f << "step_percent,exact,estimate\n";
        for (size_t t = 0; t < num_steps; ++t) {
            f << (t + 1) * STEP_PERCENT << "," << all_exact[0][t] << "," << all_estimates[0][t] << "\n";
        }
    } {
        std::ofstream f("data/" + prefix + "_graph2.csv");
        f << "step_percent,mean_exact,mean_estimate,std_estimate,upper,lower\n";
        for (size_t t = 0; t < num_steps; ++t) {
            int pct = (t + 1) * STEP_PERCENT;
            f << pct << ","
                    << mean_ex[t] << ","
                    << mean_est[t] << ","
                    << std_est[t] << ","
                    << mean_est[t] + std_est[t] << ","
                    << mean_est[t] - std_est[t] << "\n";
        }
    }

    {
        std::ofstream f("data/" + prefix + "_analysis.csv");
        f << "step_percent,mean_exact,mean_estimate,rel_error_pct,std_estimate,rel_std_pct\n";
        for (size_t t = 0; t < num_steps; ++t) {
            int pct = (t + 1) * STEP_PERCENT;
            double rel_err = std::abs(mean_est[t] - mean_ex[t]) / mean_ex[t] * 100.0;
            double rel_std = std_est[t] / mean_ex[t] * 100.0;
            f << pct << ","
                    << mean_ex[t] << ","
                    << mean_est[t] << ","
                    << rel_err << ","
                    << std_est[t] << ","
                    << rel_std << "\n";
        }
    }

    size_t M = 1u << B;
    double th1 = 1.042 / std::sqrt(static_cast<double>(M));
    double th2 = 1.3 / std::sqrt(static_cast<double>(M));

    std::cout << "\nB=" << B << ", M=" << M << "\n";
    std::cout << "Теория: 1.042/sqrt(M) = " << th1 * 100 << "%\n";
    std::cout << "Теория: 1.3/sqrt(M)   = " << th2 * 100 << "%\n\n";

    printf("Шаг%%  | E(F0t)    | E(Nt)     | Отн.ош.E  | sigma(Nt) | Отн.sigma\n");
    printf("------+-----------+-----------+-----------+-----------+----------\n");
    for (size_t t = 0; t < num_steps; ++t) {
        int pct = (t + 1) * STEP_PERCENT;
        double rel_err = std::abs(mean_est[t] - mean_ex[t]) / mean_ex[t] * 100.0;
        double rel_std = std_est[t] / mean_ex[t] * 100.0;
        printf("%3d%%  | %9.1f | %9.1f | %8.3f%%  | %9.1f | %7.3f%%\n",
               pct, mean_ex[t], mean_est[t], rel_err, std_est[t], rel_std);
    }
    std::cout << "\n";
}

int main() {
    const size_t STREAM_SIZE = 100000;
    const size_t NUM_STREAMS = 30;
    const int STEP_PERCENT = 10;

    run_experiment(14, STREAM_SIZE, NUM_STREAMS, STEP_PERCENT, "B14");

    run_experiment(4, STREAM_SIZE, NUM_STREAMS, STEP_PERCENT, "B4");

    run_experiment(8, STREAM_SIZE, NUM_STREAMS, STEP_PERCENT, "B8");
    run_experiment(10, STREAM_SIZE, NUM_STREAMS, STEP_PERCENT, "B10");

    return 0;
}
