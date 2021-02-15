import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


def arg_setup():
    parser = argparse.ArgumentParser(description="DNA-motifs-oris - A set of DNA tools for locating ORIs and motifs")
    parser.add_argument("--input", "-i", type=str, required=False,
                        help="File containing the inputs parameters of the algorithm you wish to run")
    parser.add_argument("--pattern-count", "-pc", action="store_true",
                        help="Counts the number of times a pattern appears on a dna sequence")
    parser.add_argument("--approximate-pattern-count", "-apc", action="store_true",
                        help="Counts the number of times a pattern appears on a dna sequence with as maximum "
                             "as d mismatches")
    parser.add_argument("--neighbors", "-n", action="store_true",
                        help="Find the d-neighborhood of a string with a Hamming Distance of d as maximum")
    parser.add_argument("--computing-frequencies", "-cf", action="store_true",
                        help="Generates a frequency array for every single k-mer pattern we find in a DNA sequence")
    parser.add_argument("--computing-frequencies-mismatches", "-cfm", action="store_true",
                        help="Computes a frequency array with up to d mismatches")
    parser.add_argument("--frequent-word", "-fw", action="store_true",
                        help="Finds the most frequent k-mer word in a DNA sequence")
    parser.add_argument("--faster-frequent-word", "-ffw", action="store_true",
                        help="Finds the most frequent k-mer word in a DNA sequence (Faster than Frequent Word "
                             "algorithm when k is small)")
    parser.add_argument("--frequent-word-sorting", "-fws", action="store_true",
                        help="Finds the most frequent k-mer word in a DNA sequence (Faster than -ffw and -fw even when "
                             "k is large")
    parser.add_argument("--frequent-word-mismatches", "-fwm", action="store_true",
                        help="Finds the most frequent k-mers with mismatches in a string")
    parser.add_argument("--frequent-word-mismatches-reverse-complement", "-fwmrc", action="store_true",
                        help="Finds the most frequent k-mers with mismatches in a string and its reverse complements")
    parser.add_argument("--reverse-complement", "-rc", action="store_true",
                        help="Computes the complement strand of a given DNA strand sequence in 5'->3' direction")
    parser.add_argument("--pattern-matching", "-pm", action="store_true",
                        help="Finds the indices where the pattern appears in the dna strand")
    parser.add_argument("--approximate-pattern-matching", "-apm", action="store_true",
                        help="Find all approximate occurrences of a pattern in a string that doesn't "
                             "overpass a certain number of mismatches")
    parser.add_argument("--pattern-to-number", "-p2n", action="store_true",
                        help="Retrieves the index of the k-mer pattern from the lexicographical array")
    parser.add_argument("--number-to-pattern", "-n2p", action="store_true",
                        help="Retrieves the DNA pattern associated with a specific index and k-mer length")
    parser.add_argument("--clump-finding", "-cfa", action="store_true",
                        help="Find patterns forming clumps in a DNA sequence")
    parser.add_argument("--faster-clump-finding", "-fcfa", action="store_true",
                        help="Find patterns forming clumps in a DNA sequence (Faster than -cfa)")
    parser.add_argument("--skew", "-s", action="store_true",
                        help="Records the difference count between the number of G and C in a linear DNA strand")
    parser.add_argument("--maximum-skew", "-Ms", action="store_true",
                        help="Locates the position in a genome where the skew diagram attains a maximum")
    parser.add_argument("--minimum-skew", "-ms", action="store_true",
                        help="Locates the position in a genome where the skew diagram attains a minimum")
    parser.add_argument("--hamming-distance", "-hd", action="store_true",
                        help="Calculates the Hamming Distance between two strings of the same length")
    parser.add_argument("--motif-enumeration", "-me", action="store_true",
                        help="Finds all motifs of a k-mer pattern with d mismatches in a DNA string")
    parser.add_argument("--distance-pattern-strings", "-dps", action="store_true",
                        help="Calculates the sum of the distances between a pattern and a set of DNA strings")
    parser.add_argument("--median-string-algorithm", "-msa", action="store_true",
                        help="Find a median motif in a set of DNA strings")
    parser.add_argument("--profile-most-probable-kmer", "-pmpk", action="store_true",
                        help="Find a profile-most probable kmer in a DNA string")
    parser.add_argument("--greedy-motif-search", "-gms", action="store_true",
                        help="Computes the best motifs of a set of DNA strings based on a greedy algorithm "
                             "(Deprecated).")
    parser.add_argument("--greedy-motif-search-corrected", "-gmsc", action="store_true",
                        help="Computes the best motifs of a set of DNA strings based on a greedy algorithm."
                             "Applies LaPlace's Rule of Succession to resolve the zero probability computation when"
                             "estimating the consensus pattern")
    parser.add_argument("--randomized-motif-search", "-rms", type=int,
                        help="Computes the best motifs of a set of DNA strings by applying the Monte Carlo algorithm")
    parser.add_argument("--gibbs-sampler", "-gs", type=int,
                        help="Computes the best motifs of a set of DNA strings by apply the Gibbs Sampler algorithm")
    return parser.parse_args()


sorted_bases = \
        {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3
        }


def pattern_count(text, pattern):
    count = 0
    for i in range(len(text)-len(pattern)+1):
        if text[i: i + len(pattern)] == pattern:
            count += 1
    return count


def approximate_pattern_count(pattern, text, d):
    count = 0
    for i in range(len(text)-len(pattern)+1):
        substr = text[i:i+len(pattern)]
        if hamming_distance(substr, pattern) <= d:
            count += 1

    return count


def neighbors(pattern, d):
    if d == 0:
        return {pattern}

    if len(pattern) == 1:
        return {'A', 'C', 'G', 'T'}

    neighborhood = set()
    suffix_neighbors = neighbors(pattern[1:], d)
    for text in suffix_neighbors:
        if hamming_distance(pattern[1:], text) < d:
            for x in ['A', 'C', 'G', 'T']:
                neighborhood.add(x + text)
        else:
            neighborhood.add(pattern[0] + text)

    return sorted(neighborhood)


def computing_frequencies(text, k):
    frequencies = [0] * 4**k
    for i in range(len(text)-k+1):
        pattern = text[i:i+k]
        j = pattern_to_number(pattern)
        frequencies[j] += 1

    return frequencies


def computing_frequencies_mismatches(text, k, d):
    frequencies = [0] * 4 ** k
    for i in range(len(text)-k+1):
        pattern = text[i:i+k]
        neighborhood = neighbors(pattern, d)
        for neighbor in neighborhood:
            j = pattern_to_number(neighbor)
            frequencies[j] += 1

    return frequencies


def frequent_word(text, k):
    count = list()
    for i in range(len(text)-k):
        pattern = text[i:i+k]
        count.append(pattern_count(text, pattern))

    max_count = max(count)
    max_idx = np.where(np.array(count) == max_count)[0]

    return sorted({text[idx: idx + k] for idx in max_idx})


def faster_frequent_word(text, k):
    frequent_patterns = set()
    frequencies = computing_frequencies(text, k)
    max_count = max(frequencies)
    for i in range(4**k):
        if frequencies[i] == max_count:
            pattern = number_to_pattern(i, k)
            frequent_patterns.add(pattern)
    return sorted(frequent_patterns)


def frequent_word_sorting(text, k):
    frequent_patterns = set()
    idx = list()
    count = list()
    for i in range(len(text)-k+1):
        pattern = text[i:i+k]
        idx.append(pattern_to_number(pattern))
        count.append(1)

    sorted_idx = sorted(idx)
    for i in range(1, len(text)-k+1):
        if sorted_idx[i] == sorted_idx[i-1]:
            count[i] = count[i-1] + 1

    max_count = max(count)
    for i in range(len(text)-k+1):
        if count[i] == max_count:
            pattern = number_to_pattern(sorted_idx[i], k)
            frequent_patterns.add(pattern)

    return frequent_patterns


def frequent_word_mismatches(text, k, d):
    frequent_patterns = set()
    frequencies = computing_frequencies_mismatches(text, k, d)
    max_count = max(frequencies)
    for i in range(4**k):
        if frequencies[i] == max_count:
            pattern = number_to_pattern(i, k)
            frequent_patterns.add(pattern)
    return sorted(frequent_patterns)


def frequent_word_mismatches_reverse_complement(text, k, d):
    frequent_patterns = set()
    text += reverse_complement(text)
    frequencies = computing_frequencies_mismatches(text, k, d)
    max_count = max(frequencies)
    for i in range(4**k):
        if frequencies[i] == max_count:
            pattern = number_to_pattern(i, k)
            frequent_patterns.add(pattern)
    return sorted(frequent_patterns)


def reverse_complement(text):
    comp_nuc = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C"
    }

    reverse_strand = ""
    for p in text:
        reverse_strand += comp_nuc[p]

    return reverse_strand[::-1]


def pattern_matching(pattern, genome):
    matches = list()
    for i in range(len(genome)-len(pattern)+1):
        substr = genome[i:i+len(pattern)]
        if substr == pattern:
            matches.append(i)

    return matches


def approximate_pattern_matching(pattern, genome, d):
    matches = list()
    k = len(pattern)

    for i in range(len(genome)-k+1):
        substr = genome[i:i+k]
        if hamming_distance(substr, pattern) <= d:
            matches.append(i)

    return matches


def pattern_to_number(pattern):
    k = len(pattern)
    idx = 0
    for c in pattern:
        k -= 1
        idx += 4 ** k * sorted_bases[c]

    return idx


def number_to_pattern(idx, k):
    p = idx
    pattern = ""
    for _ in range(k-1):
        q = p // 4
        pattern += list(sorted_bases.keys())[p % 4]
        p = q

    pattern += list(sorted_bases.keys())[p]
    return pattern[::-1]


def clump_finding(genome, k, L, t):
    frequent_patterns = set()
    clump = [0] * (4**k)

    for i in range(len(genome)-L+1):
        text = genome[i:i+L]
        frequencies = computing_frequencies(text, k)
        for j in range(4**k):
            if frequencies[j] >= t:
                clump[j] = 1

    for i in range(4**k):
        if clump[i] == 1:
            pattern = number_to_pattern(i, k)
            frequent_patterns.add(pattern)

    return sorted(frequent_patterns)


def faster_clump_finding(genome, k, L, t):
    frequent_patterns = set()
    clump = [0] * (4**k)

    text = genome[0:L]
    frequencies = computing_frequencies(text, k)
    for i in range(4**k):
        if frequencies[i] >= t:
            clump[i] = 1

    for i in range(1, len(genome)-L+1):
        first_pattern = genome[i-1:i+k-1]
        index = pattern_to_number(first_pattern)
        frequencies[index] -= 1
        last_pattern = genome[i+L-k:i+L]
        index = pattern_to_number(last_pattern)
        frequencies[index] += 1

        if frequencies[index] >= t:
            clump[index] = 1

    for i in range(4**k):
        if clump[i] == 1:
            pattern = number_to_pattern(i, k)
            frequent_patterns.add(pattern)

    return sorted(frequent_patterns)


def skew(genome):
    skew_seq = [0]

    for i, nucleotide in enumerate(genome):
        if nucleotide == 'C':
            skew_seq.append(skew_seq[i]-1)
        elif nucleotide == 'G':
            skew_seq.append(skew_seq[i]+1)
        else:
            skew_seq.append(skew_seq[i])

    return skew_seq


def maximum_skew(genome):
    skew_seq = skew(genome)
    min_value = max(skew_seq)
    idx = list()

    for i, value in enumerate(skew_seq):
        if value == min_value:
            idx.append(i)
    return idx


def minimum_skew(genome):
    skew_seq = skew(genome)
    min_value = min(skew_seq)
    idx = list()

    for i, value in enumerate(skew_seq):
        if value == min_value:
            idx.append(i)
    return idx


def hamming_distance(p, q):
    if len(p) != len(q):
        return -1

    distance = 0
    for i in range(len(p)):
        if p[i] != q[i]:
            distance += 1

    return distance


def motif_enumeration(dna, k, d):
    k_mers = [set() for _ in dna]
    for pos, pattern in enumerate(dna):
        for k_pos in range(len(pattern)-k+1):
            for neighbor in neighbors(pattern[k_pos:k_pos+k], d):
                k_mers[pos].add(neighbor)

    patterns = k_mers[0]
    for k_set in k_mers:
        patterns = patterns & k_set

    return patterns


def distance_pattern_strings(pattern, dna):
    k = len(pattern)
    distance = 0
    for text in dna:
        h_distance = 1000
        for k_pos in range(len(text)-k+1):
            t_pattern = text[k_pos:k_pos+k]
            distance_neighbor = hamming_distance(pattern, t_pattern)
            if h_distance > distance_neighbor:
                h_distance = distance_neighbor

        distance += h_distance

    return distance


def median_string_algorithm(dna, k):
    distance = 1000
    for i in range(4**k):
        pattern = number_to_pattern(i, k)
        d_pattern_dna = distance_pattern_strings(pattern, dna)
        if distance > d_pattern_dna:
            distance = d_pattern_dna
            median = pattern

    return median


def profile_most_probable_kmer(text, k, profile):
    best_probability = 0
    best_pattern = text[:k]
    for i in range(len(text)-k+1):
        pattern = text[i:i+k]
        probability = 1
        for j, base in enumerate(pattern):
            probability *= profile[base][j]

        if probability > best_probability:
            best_probability = probability
            best_pattern = pattern

    return best_pattern


def create_count_matrix(motifs, k):
    profile = np.zeros((k, 4))
    n = len(motifs)
    for i in range(n):
        motif = motifs[i]
        for j in range(k):
            if motif[j] == 'A':
                profile[j, 0] += 1.
            elif motif[j] == 'C':
                profile[j, 1] += 1.
            elif motif[j] == 'G':
                profile[j, 2] += 1.
            elif motif[j] == 'T':
                profile[j, 3] += 1.

    return profile


def score(motifs, consensus):
    score = 0
    for motif in motifs:
        score += hamming_distance(motif, consensus)

    return score


def greedy_motif_search(dna, k, t):
    n = len(dna[0])
    best_motifs = [text[:k] for text in dna]
    motifs = list()
    score_best_motifs = 1000
    for i in range(n-k+1):
        motifs.append(dna[0][i:i+k])
        for j in range(1, t):
            profile = create_count_matrix(motifs, k) / (j + 1)
            profile_dict = {k: v for k, v in zip(list(sorted_bases.keys()), profile.swapaxes(0, 1))}
            motifs.append(profile_most_probable_kmer(dna[j], k, profile_dict))

        consensus = "".join([list(sorted_bases)[x] for x in profile.swapaxes(0, 1).argmax(axis=0)])
        score_motifs = score(motifs, consensus)
        if score_motifs < score_best_motifs:
            score_best_motifs = score_motifs
            best_motifs = motifs
        motifs = list()

    return best_motifs


def greedy_motif_search_corrected(dna, k, t):
    n = len(dna[0])
    best_motifs = [text[:k] for text in dna]
    motifs = list()
    score_best_motifs = 1000
    for i in range(n - k + 1):
        motifs.append(dna[0][i:i + k])
        for j in range(1, t):
            profile = (create_count_matrix(motifs, k) + 1) / (j + 1)
            profile_dict = {k: v for k, v in zip(list(sorted_bases.keys()), profile.T)}
            motifs.append(profile_most_probable_kmer(dna[j], k, profile_dict))

        consensus = "".join([list(sorted_bases)[x] for x in profile.T.argmax(axis=0)])
        score_motifs = score(motifs, consensus)
        if score_motifs < score_best_motifs:
            score_best_motifs = score_motifs
            best_motifs = motifs
        motifs = list()

    return best_motifs


def randomized_motif_search(dna, k, t):
    n = len(dna[0])
    best_motifs = [text[i:i+k] for i, text in zip(np.random.choice(n-k+1, t), dna)]
    motifs = best_motifs.copy()

    while True:
        profile = (create_count_matrix(motifs, k) + 1) / t
        profile_dict = {k: v for k, v in zip(list(sorted_bases.keys()), profile.T)}
        for i in range(t):
            motifs[i] = profile_most_probable_kmer(dna[i], k, profile_dict)

        consensus = "". join([list(sorted_bases)[x] for x in profile.T.argmax(axis=0)])

        if score(motifs, consensus) < score(best_motifs, consensus):
            best_motifs = motifs
        else:
            return best_motifs, score(best_motifs, consensus)


def gibbs_sampler(dna, k, t, n):
    N = len(dna[0])
    best_motifs = [text[i:i+k] for i, text in zip(np.random.choice(N-k+1, t), dna)]
    motifs = best_motifs.copy()
    for j in range(1, n):
        i = np.random.choice(t)
        profile = (create_count_matrix([x for j, x in enumerate(motifs) if j != i], k) + 1) / t
        profile_dict = {k: v for k, v in zip(list(sorted_bases.keys()), profile.T)}
        motifs[i] = profile_most_probable_kmer(dna[i], k, profile_dict)

        consensus = "".join([list(sorted_bases)[x] for x in profile.T.argmax(axis=0)])

        if score(motifs, consensus) < score(best_motifs, consensus):
            best_motifs = motifs

    return best_motifs, score(best_motifs, consensus)


def main():
    np.random.seed(2017)
    parser = arg_setup()

    try:
        if parser.input is not None:
            f = open(parser.input, 'r')
            args = f.read().split('\n')

        if parser.pattern_count:
            print(pattern_count(args[0], args[1]))
        elif parser.approximate_pattern_count:
            print(approximate_pattern_count(args[0], args[1], int(args[2])))
        elif parser.neighbors:
            neighbors_array = neighbors(args[0], int(args[1]))
            print("Total number of neighbors: %d" % len(neighbors_array))
            for x in neighbors_array:
                print(x, end=" ")
        elif parser.computing_frequencies:
            for x in computing_frequencies(args[0], int(args[1])):
                print(x, end=" ")
        elif parser.computing_frequencies_mismatches:
            for x in computing_frequencies_mismatches(args[0], args[1], int(args[2])):
                print(x, end=" ")
        elif parser.frequent_word:
            for x in frequent_word(args[0], int(args[1])):
                print(x, end=" ")
        elif parser.faster_frequent_word:
            for x in faster_frequent_word(args[0], int(args[1])):
                print(x, end=" ")
        elif parser.frequent_word_sorting:
            for x in frequent_word_sorting(args[0], int(args[1])):
                print(x, end=" ")
        elif parser.frequent_word_mismatches:
            args2 = args[1].split()
            for x in frequent_word_mismatches(args[0], int(args2[0]), int(args2[1])):
                print(x, end=" ")
        elif parser.frequent_word_mismatches_reverse_complement:
            args2 = args[1].split()
            for x in frequent_word_mismatches_reverse_complement(args[0], int(args2[0]), int(args2[1])):
                print(x, end=" ")
        elif parser.reverse_complement:
            print(reverse_complement(args[0]))
        elif parser.pattern_matching:
            for x in pattern_matching(args[0], args[1]):
                print(x, end=" ")
        elif parser.approximate_pattern_matching:
            for x in approximate_pattern_matching(args[0], args[1], int(args[2])):
                print(x, end=" ")
        elif parser.pattern_to_number:
            print(pattern_to_number(args[0]))
        elif parser.number_to_pattern:
            print(number_to_pattern(int(args[0]), int(args[1])))
        elif parser.clump_finding:
            args2 = args[1].split(" ")
            clumps = clump_finding(args[0], int(args2[0]), int(args2[1]), int(args2[2]))
            print("Found %d clumps" % len(clumps))
            for x in clumps:
                print(x, end=" ")
        elif parser.faster_clump_finding:
            args2 = args[1].split(" ")
            clumps = faster_clump_finding(args[0], int(args2[0]), int(args2[1]), int(args2[2]))
            print("Found %d clumps" % len(clumps))
            for x in clumps:
                print(x, end=" ")
        elif parser.skew:
            skew_seq = skew(args[0])
            for x in skew_seq:
                print(x, end=" ")
                plt.plot(skew_seq)
                plt.show()
        elif parser.maximum_skew:
            for x in maximum_skew(args[0]):
                print(x, end=" ")
        elif parser.minimum_skew:
            for x in minimum_skew(args[0]):
                print(x, end=" ")
        elif parser.hamming_distance:
            print(hamming_distance(args[0], args[1]))
        elif parser.motif_enumeration:
            args1 = args[0].split()
            k = int(args1[0])
            d = int(args1[1])
            dna = [x for x in args[1:-1]]
            for x in motif_enumeration(dna, k, d):
                print(x, end=" ")
        elif parser.distance_pattern_strings:
            print(distance_pattern_strings(args[0], args[1].split()))
        elif parser.median_string_algorithm:
            print(median_string_algorithm(args[1:-1], int(args[0])))
        elif parser.profile_most_probable_kmer:
            profile = dict()
            profile['A'] = [float(x) for x in args[2].split()]
            profile['C'] = [float(x) for x in args[3].split()]
            profile['G'] = [float(x) for x in args[4].split()]
            profile['T'] = [float(x) for x in args[5].split()]
            print(profile_most_probable_kmer(args[0], int(args[1]), profile))
        elif parser.greedy_motif_search:
            args1 = args[0].split()
            for x in greedy_motif_search(args[1:-1], int(args1[0]), int(args1[1])):
                print(x, end=" ")
        elif parser.greedy_motif_search_corrected:
            args1 = args[0].split()
            for x in greedy_motif_search_corrected(args[1:-1], int(args1[0]), int(args1[1])):
                print(x, end=" ")
        elif parser.randomized_motif_search is not None:
            args1 = args[0].split()
            best_motifs, score_best_motifs = randomized_motif_search(args[1:-1], int(args1[0]), int(args1[1]))
            for _ in range(parser.randomized_motif_search):
                motifs, score_motifs = randomized_motif_search(args[1:-1], int(args1[0]), int(args1[1]))

                if score_motifs < score_best_motifs:
                    best_motifs = motifs
                    score_best_motifs = score_motifs

            for x in best_motifs:
                print(x)
        elif parser.gibbs_sampler is not None:
            args1 = args[0].split()
            best_motifs, score_best_motifs = gibbs_sampler(args[1:-1], int(args1[0]), int(args1[1]),int(args1[2]))
            for _ in range(parser.gibbs_sampler):
                motifs, score_motifs = gibbs_sampler(args[1:-1], int(args1[0]), int(args1[1]), int(args1[2]))

                if score_motifs < score_best_motifs:
                    best_motifs = motifs
                    score_best_motifs = score_motifs

            for x in best_motifs:
                print(x)

    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
