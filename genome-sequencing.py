import copy
import argparse
import numpy as np
from collections import Counter


class WriteOutput(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        f = open(values, 'a')
        f.truncate(0)
        setattr(namespace, self.dest, open(values, 'a'))


def arg_setup():
    parser = argparse.ArgumentParser(description="DNA-motifs-oris - A set of DNA tools for locating ORIs and motifs")
    parser.add_argument("input", nargs='?',
                        help="File containing the inputs parameters of the algorithm you wish to run")
    parser.add_argument("--output", "-o", type=str, action=WriteOutput)
    parser.add_argument("--integer-mass-table", "-imt", type=str,
                        help="Loads integer mass table for all the 20 peptides")
    parser.add_argument('--kmer-composition', '-kc', action='store_true',
                        help='A collection of all k-mer substrings of a DNA text (included repeated k-mers)')
    parser.add_argument("-path-to-genome", "-p2g", action='store_true',
                        help='Reconstructs the genome from a genome path of k-mer patterns')
    parser.add_argument("--overlap-graph", "-og", action='store_true',
                        help='Constructs the overlapping graph of a set of k-mer composition of a certain DNA pattern')
    parser.add_argument("--de-Bruijn-graph", "-dBg", action='store_true',
                        help='Construct the de Bruijn graph of k-mers given a certain DNA pattern')
    parser.add_argument("--de-Bruijn-graph-kmers", "-dBgk", action='store_true',
                        help='Construct the de Bruijn graph given a set of k-mers')
    parser.add_argument("--Eulerian-cycle", "-Ec", action="store_true",
                        help="Returns an Eulerian cycle out of a graph")
    parser.add_argument("--Eulerian-path", "-Ep", action="store_true",
                        help="Return an Eulerian path out of a graph")
    parser.add_argument("--genome-reconstruction", "-gr", action="store_true",
                        help="Reconstructs the genome out of chunks of DNA by generating a de Bruijn graph and finding"
                             "the Eulerian path from it")
    parser.add_argument("--k-universal-circular-string", "-kucs", action="store_true",
                        help="Computes the k-Universal circular string compounded of every possible k-mer")
    parser.add_argument("--string-spelled-gapped-pattern", "-ssgp", action="store_true",
                        help="Constructs strings of prefixes and suffixes and checks whether the have "
                             "a perfect overlap")
    parser.add_argument("--reconstruct-read-pairs", "-rrp", action="store_true",
                        help="Reconstructs a DNA strings from read-pairs")
    parser.add_argument("--maximal-non-branching-path", "-mnbp", action="store_true",
                        help="Generates all non-branching path in a graph")
    parser.add_argument("--contig-generator", "-cg", action="store_true",
                        help="Generates the contigs from a collection of reads with imperfect coverage")
    parser.add_argument("--translate-RNA", "-tRNA", action="store_true",
                        help="Translate an RNA string into an amino acid string")
    parser.add_argument("--peptide-encoding", "-pe", action="store_true",
                        help="Finds substrings of a genome encoding a given amino acid sequence")
    parser.add_argument("--number-of-dna-strings-translate-peptide", "-nodstp", action="store_true",
                        help="Counts the total number of DNA strings which translates into a certain peptide sequence")
    parser.add_argument("--number-of-subpeptides-from-cyclic-peptide", "-nosfcp", action="store_true",
                        help="Calculates how many subpeptides does a cyclic peptide of length n have")
    parser.add_argument("--number-of-subpeptides-from-linear-peptide", "-nosflp", action="store_true",
                        help="Calculates how many subpeptides does a linear peptide of length n have")
    parser.add_argument("--linear-theoretical-spectrum", "-lts", action="store_true",
                        help="Calculates the theoretical spectrum of a linear protein")
    parser.add_argument("--cyclic-theoretical-spectrum", "-cts", action="store_true",
                        help="Calculates the theoretical spectrum of a cyclic protein")
    parser.add_argument("--brute-force-cyclopeptide-sequencing", "-bfcps", action="store_true",
                        help="Generates all possible peptides whose mass is equal to the total peptide mass")
    parser.add_argument("--cyclopeptide-sequencing", "-cps", action="store_true",
                        help="Generates consistent peptides with the experimental spectrum "
                             "whose mass is equal to the total peptide mass")
    parser.add_argument("--cyclopeptide-scoring", "-cs", action="store_true",
                        help="Computes the score of a cyclic peptide against a spectrum")
    parser.add_argument("--leaderboard-cyclopeptide-sequencing", "-lcps", action="store_true",
                        help="Sorts the peptides candidates with descending score value")
    parser.add_argument("--linear-peptide-scoring", "-lps", action="store_true",
                        help="Computes the score of a linear peptide with respecto to a spectrum")
    parser.add_argument("--trim-leaderboard", "-tl", action="store_true",
                        help="Retains the top N scoring peptides including ties")
    parser.add_argument("--find-maximum-score-leaderboard-cyclopeptide-sequencing", "-fmslcps", action="store_true",
                        help="Computes all the candidate peptides which total mass is equal to the spectra mass and "
                             "within a certain maximum score")
    parser.add_argument("--spectral-convolution", "-sc", action="store_true",
                        help="Computes the convolution of a spectrum")
    parser.add_argument("--convolution-cyclopeptide-sequencing", "-ccps", action="store_true",
                        help="Computes a peptide with the top M elements of the convolution of spectrum that fall "
                             "bewtween 57 and 200 amd where the size of leaderboard is restricted to the top N")
    return parser.parse_args()


comp_nuc = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C"
    }

genetic_code = {
    "AAA": "K", "CAA": "Q", "GAA": "E", "UAA": "*",
    "AAC": "N", "CAC": "H", "GAC": "D", "UAC": "Y",
    "AAG": "K", "CAG": "Q", "GAG": "E", "UAG": "*",
    "AAU": "N", "CAU": "H", "GAU": "D", "UAU": "Y",
    "ACA": "T", "CCA": "P", "GCA": "A", "UCA": "S",
    "ACC": "T", "CCC": "P", "GCC": "A", "UCC": "S",
    "ACG": "T", "CCG": "P", "GCG": "A", "UCG": "S",
    "ACU": "T", "CCU": "P", "GCU": "A", "UCU": "S",
    "AGA": "R", "CGA": "R", "GGA": "G", "UGA": "*",
    "AGC": "S", "CGC": "R", "GGC": "G", "UGC": "C",
    "AGG": "R", "CGG": "R", "GGG": "G", "UGG": "W",
    "AGU": "S", "CGU": "R", "GGU": "G", "UGU": "C",
    "AUA": "I", "CUA": "L", "GUA": "V", "UUA": "L",
    "AUC": "I", "CUC": "L", "GUC": "V", "UUC": "F",
    "AUG": "M", "CUG": "L", "GUG": "V", "UUG": "L",
    "AUU": "I", "CUU": "L", "GUU": "V", "UUU": "F"
}


def cmd_input(args=[]):
    while True:
        try:
            line = input()
        except EOFError:
            break
        args.append(line)

    return args


def input_graph(args):
    data = [x.split(' -> ') for x in args]
    from_nodes = [x[0] for x in data]
    to_nodes = [x[1].split(',') for x in data]

    return {k: v for k, v in zip(from_nodes, to_nodes)}


def input_integer_mass_table(file):
    f = open(file)
    return {k: int(v) for k, v in [x.split() for x in f.read().split('\n') if x != ""]}


def input_list(args, typefun):
    return [typefun(x) for x in args.split()]


def print_text(text):
    if parser.output is not None:
        parser.output.write(str(text))
    print(text)


def print_list(out, delimiter='\n'):
    for i, x in enumerate(out):
        if parser.output is not None:
            if isinstance(x, int):
                parser.output.write(str(x))
            else:
                parser.output.write(x)
            if i < len(out) - 1:
                parser.output.write(delimiter)
        print(x, end=delimiter)


def print_graph(graph_list):
    text = ""
    for j, (k, v) in enumerate(graph_list.items()):
        text += k + " -> "
        for i in range(len(v)-1):
            text += v[i] + ", "

        text += v[-1]
        if j != len(graph_list) - 1:
            text += '\n'

    if parser.output is not None:
        parser.output.write(text)
    print(text)


def print_path(path):
    text = ""
    for i in range(len(path)-1):
        text += path[i] + " -> "

    if len(path) > 0:
        text += path[-1]

    if parser.output is not None:
        parser.output.write(text + '\n')
    print(text)


def reverse_complement(text):
    reverse_strand = ""
    for p in text:
        reverse_strand += comp_nuc[p]

    return reverse_strand[::-1]


def kmer_composition(k, text):
    return sorted([text[i:i+k] for i in range(len(text)-k+1)])


def path_to_genome(path):
    genome = path[0]
    for i in range(len(path)-1):
        if path[i][1:] == path[i+1][:-1]:
            genome += path[i+1][-1]
    return genome


def overlap_graph(patterns):
    n = len(patterns)
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            graph[i, j] = 1 if patterns[i][1:] == patterns[j][:-1] else 0

    graph_list = adjacency_list(graph, patterns)
    return graph, graph_list


def adjacency_list(graph, patterns):
    graph_list = dict()
    for i, pattern in enumerate(patterns):
        pattern_idx = np.where(graph[i] > 0)[0]
        if len(pattern_idx) > 0:
            for idx in pattern_idx:
                values = [patterns[i] for i in pattern_idx] * int(graph[i, idx])
            graph_list.update({pattern: values})
            return graph_list


def de_bruijn_graph(k, text):
    patterns = kmer_composition(k, text)
    patterns_red = np.array(sorted(set(kmer_composition(k-1, text))))
    n = len(patterns_red)
    graph = np.zeros((n,n))
    for i in range(len(patterns)):
        for j in range(n):
            idx = np.where(patterns_red == patterns[i][:-1])
            graph[idx, j] += 1 if patterns[i][1:] == patterns_red[j] else 0
    graph_list = adjacency_list(graph, patterns_red)
    return graph, graph_list


def fast_de_bruijn_graph(k, text):
    patterns = kmer_composition(k, text)
    graph = dict()
    for pattern in patterns:
        prefix = pattern[:-1]
        suffix = pattern[1:]
        if prefix not in list(graph.keys()):
            graph[prefix] = [suffix]
        else:
            graph[prefix].append(suffix)

    return graph


def de_bruijn_graph_kmers(patterns):
    graph = dict()
    for pattern in patterns:
        prefix = pattern[:-1]
        suffix = pattern[1:]
        if prefix not in graph.keys():
            graph[prefix] = [suffix]
        else:
            graph[prefix].append(suffix)

    return graph


def eulerian_cycle(graph):
    backtrack = {k: [len(v), v[:]] for k, v in graph.items()}
    cycle = list()
    start_node = np.random.choice(list(graph))
    node_track = [start_node]
    while len(node_track) > 0:
        curr_node = node_track[-1]

        if curr_node not in cycle and curr_node in backtrack.keys() and backtrack[curr_node][0] > 0:
            next_node = np.random.choice(backtrack[curr_node][1])
            next_node_idx = backtrack[curr_node][1].index(next_node)
            backtrack[curr_node][1].pop(next_node_idx)
            backtrack[curr_node][0] -= 1
            node_track.append(next_node)
        else:
            cycle.append(curr_node)
            node_track.pop()
            if curr_node in backtrack.keys():
                del backtrack[curr_node]
    if cycle[0] == cycle[-1]:
        cycle.reverse()
        return cycle
    return list()


def eulerian_grader(graph):
    graph_for_method = copy.deepcopy(graph)
    path_list = eulerian_cycle(graph_for_method)
    grade = []
    for index in range(len(path_list) - 1):
        elem = path_list[index]
        next_elem = path_list[index + 1]
        if elem not in graph.keys():
            print("invalid elem: " + str(elem) + " at posn " + str(index))
            grade.append("src")
        elif next_elem not in graph[elem]:
            print("invalid edge: " + str(elem) + " to " + str(next_elem))
            grade.append("dst")
        else:
            graph[elem].remove(next_elem)
            if not graph[elem]:
                del graph[elem]
            grade.append("OK")
    if not graph:
        print("emptied!")
    else:
        print(graph)
    return grade


def indegrees(graph):
    in_edges = dict()
    for v in graph.values():
        for elem in v:
            if elem not in in_edges.keys():
                in_edges[elem] = 1
            else:
                in_edges[elem] += 1

    nodes_no_indeg = dict()
    for k in graph.keys():
        if k not in in_edges.keys():
            nodes_no_indeg[k] = 0
    in_edges.update(nodes_no_indeg)
    return in_edges


def outdegrees(graph):
    out_edges = dict()
    for k, v in graph.items():
        out_edges.update({k: len(v)})

    nodes_no_outdeg = dict()
    for values in graph.values():
        for val in values:
            if val not in out_edges.keys():
                nodes_no_outdeg.update({val: 0})
    out_edges.update(nodes_no_outdeg)
    return out_edges


def eulerian_path_limit(in_edges, out_edges):
    start_node = list()
    for k in in_edges.keys():
        if out_edges[k] - in_edges[k] == 1:
            start_node.append(k)

    if len(start_node) > 0:
        start_node = np.random.choice(start_node)
    else:
        start_node = np.random.choice(list(in_edges.keys()))

    return start_node


def eulerian_path(graph):
    in_edges = indegrees(graph)
    out_edges = outdegrees(graph)
    backtrack = {k: v[:] for k, v in graph.items()}
    backtrack.update({k: [] for k in in_edges.keys() if k not in graph.keys()})
    start_node = eulerian_path_limit(in_edges, out_edges)
    node_track = [start_node]
    path = list()
    while len(node_track) > 0:
        curr_node = node_track[-1]

        if curr_node in backtrack.keys() and out_edges[curr_node] > 0:
            next_node = np.random.choice(backtrack[curr_node])
            next_node_idx = backtrack[curr_node].index(next_node)
            backtrack[curr_node].pop(next_node_idx)
            out_edges[curr_node] -= 1
            node_track.append(next_node)
        else:
            path.append(curr_node)
            node_track.pop()
            if curr_node in backtrack.keys():
                del backtrack[curr_node]

    path.reverse()
    return path


def string_reconstruction(k, patterns):
    db_graph = de_bruijn_graph_kmers(patterns)
    path = eulerian_path(db_graph)
    return path_to_genome(path)


def generate_binary_pattern(n):
    if (n <= 0):
        return

    arr = list()

    arr.append("0")
    arr.append("1")

    i = 2
    j = 0
    while (True):

        if i >= 1 << n:
            break

        for j in range(i - 1, -1, -1):
            arr.append(arr[j])

        for j in range(i):
            arr[j] = "0" + arr[j]

        for j in range(i, 2 * i):
            arr[j] = "1" + arr[j]
        i = i << 1
    return arr


def k_universal_circular_string(k):
    patterns = generate_binary_pattern(k)
    db_graph = de_bruijn_graph_kmers(patterns)
    cycle = eulerian_cycle(db_graph)
    return path_to_genome(cycle[:-(k-1)])


def string_spelled_gapped_pattern(k, d, pairs):
    first_patterns = [x for x, y in pairs]
    second_patterns = [y for x, y in pairs]
    prefix_str = path_to_genome(first_patterns)
    suffix_str = path_to_genome(second_patterns)
    for i in range(k+d, len(prefix_str)):
        if prefix_str[i] != suffix_str[i-k-d]:
            return None

    return prefix_str + suffix_str[-(k+d):]


def paired_de_bruijn_graph(pairs):
    read_1 = [x for x, y in pairs]
    read_2 = [y for x, y in pairs]
    paired_graph = dict()
    for elem_1, elem_2 in zip(read_1, read_2):
        key = elem_1[:-1] + '|' + elem_2[:-1]
        value = elem_1[1:] + '|' + elem_2[1:]
        if key not in paired_graph.keys():
            paired_graph[key] = [value]
        else:
            paired_graph[key].append(value)
    return paired_graph


def reconstruct_read_pairs(k, d, pairs):
    paired_graph = paired_de_bruijn_graph(pairs)
    path = eulerian_path(paired_graph)
    path_split = [(x.split('|')[0], x.split('|')[1]) for x in path]
    return string_spelled_gapped_pattern(k, d, path_split)


def is_non_branching(v, in_edges, out_edges):
    return in_edges[v] == out_edges[v] and in_edges[v] == 1


def maximal_non_branching_paths(graph):
    paths = list()
    in_edges = indegrees(graph)
    out_edges = outdegrees(graph)
    for v in graph.keys():
        if not is_non_branching(v, in_edges, out_edges):
            if out_edges[v] > 0:
                for w in graph[v]:
                    non_branching_path = [v, w]
                    while is_non_branching(w, in_edges, out_edges):
                        u = graph[w][0]
                        non_branching_path.append(u)
                        w = u
                    paths.append(non_branching_path)

    isolated_cycle(graph, in_edges, out_edges, paths)

    return paths


def isolated_cycle(graph, in_edges, out_edges, paths):
    used = list()
    for v in graph.keys():
        cycle = list()
        if v not in used:
            start = v
            flag = False
            while is_non_branching(v, in_edges, out_edges):
                if v not in cycle:
                    cycle.append(v)
                for w in graph[v]:
                    cycle.append(w)
                    used.append(v)
                    v = w
                if v == start:
                    flag = True
                    break
            if flag:
                paths.append(cycle)


def translate_rna(rna):
    protein = ""
    for i in range(0, len(rna), 3):
        if i+3 <= len(rna):
            codon = rna[i:i+3]
            protein += genetic_code[codon]
    return protein


def transcription_dna(dna):
    rna = ""
    for nucleotide in dna:
        if nucleotide == "T":
            rna += "U"
        else:
            rna += nucleotide

    return rna


def peptide_encoding(dna, amino_pattern):
    matches = list()
    dna_len = len(dna)
    amino_len = len(amino_pattern) * 3
    for i in range(dna_len-amino_len+1):
        chunk_dna = dna[i:i+amino_len]
        if translate_rna(transcription_dna(chunk_dna)) == amino_pattern or \
                translate_rna(transcription_dna(reverse_complement(chunk_dna))) == amino_pattern:
            matches.append(chunk_dna)
    return matches


def number_of_subpeptides_from_peptide(n, is_cyclic=False):
    if is_cyclic:
        return n*(n-1)
    else:
        return int(n*(n+1)/2 + 1)


def theoretical_spectrum(peptide, is_cyclic=False):
    prefix_mass = [0]
    for i in range(len(peptide)):
        for aa in imt.keys():
            if aa == peptide[i]:
                prefix_mass.append(prefix_mass[i] + imt[aa])
                break
    if is_cyclic:
        peptide_mass = prefix_mass[len(peptide)]

    ts = [0]
    for i in range(len(peptide)):
        for j in range(i+1, len(peptide)+1):
            mass_diff = prefix_mass[j] - prefix_mass[i]
            ts.append(mass_diff)
            if is_cyclic and i > 0 and j < len(peptide):
                ts.append(peptide_mass-mass_diff)
    return sorted(ts)


def brute_force_cyclopeptide_sequencing(spectra_mass):
    # TODO: Solve this problem later
    return []


def expand(spectra_mass, candidate_peptides, n=None, with_leaderboard=False):
    expanded_peptides = set()
    for candidate in candidate_peptides:
        for aa, aa_mass in imt.items():
            peptide = candidate + aa
            if is_consistent(spectra_mass, peptide) and not with_leaderboard:
                expanded_peptides.add(peptide)
            elif mass(peptide) <= parent_mass(spectra_mass) and with_leaderboard:
                expanded_peptides.add(peptide)

    if n is not None:
        return trim_leaderboard(expanded_peptides, spectra_mass, n)
    else:
        return expanded_peptides


def is_consistent(spectra_mass, peptide, is_cyclic=False):
    mass_count = generate_mass_counter(spectra_mass)

    ts_peptide = theoretical_spectrum(peptide, is_cyclic=is_cyclic)
    for theoretical_mass in ts_peptide:
        if theoretical_mass not in mass_count.keys():
            return False

        mass_count[theoretical_mass] -= 1
        if mass_count[theoretical_mass] < 0:
            return False

    return True


def generate_mass_counter(spectra_mass):
    mass_count = {k: 0 for k in spectra_mass}
    for experimental_mass in spectra_mass:
        mass_count[experimental_mass] += 1
    return mass_count


def mass(peptide):
    return sum([imt[aa] for aa in peptide])


def parent_mass(spectra_mass):
    return spectra_mass[-1]


def is_same_spectrum(spectra_mass, peptide):
    peptide_cts = theoretical_spectrum(peptide, True)
    if len(spectra_mass) != len(peptide_cts):
        return False

    for experimental_mass, theoretical_mass in zip(spectra_mass, peptide_cts):
        if experimental_mass != theoretical_mass:
            return False

    return True


def cyclopeptide_sequencing(spectra_mass):
    candidate_peptides = {""}
    final_peptides = set()
    while len(candidate_peptides) > 0:
        candidate_peptides = expand(spectra_mass, candidate_peptides)
        for peptide in candidate_peptides:
            if mass(peptide) == parent_mass(spectra_mass):
                if is_same_spectrum(spectra_mass, peptide):
                    final_peptides.add(peptide)

    return final_peptides


def peptides_to_mass_sequence(peptides):
    mass_sequences = list()
    for peptide in peptides:
        peptide_mass_seq = ""
        for aa in peptide:
            peptide_mass_seq += str(imt[aa]) + "-"
        mass_sequences.append(peptide_mass_seq[:-1])
    return set(mass_sequences)


def number_of_dna_strings_translate_peptide(peptide):
    count = 1
    for aa in peptide:
        aa_translations = 0
        for v in genetic_code.values():
            if v == aa:
                aa_translations += 1
        count *= aa_translations
    return count


def peptide_scoring(peptide, mass_spectrum, is_cyclic=False):
    ts = theoretical_spectrum(peptide, is_cyclic)
    mass_count = generate_mass_counter(mass_spectrum)
    count = 0
    for theoretical_mass in ts:
        if theoretical_mass in mass_spectrum and mass_count[theoretical_mass] > 0:
            count += 1
            mass_count[theoretical_mass] -= 1
    return count


def generate_leaderboard(peptides, spectrum):
    leaderboard = dict()
    for peptide in peptides:
        leaderboard[peptide] = peptide_scoring(peptide, spectrum)
    leaderboard = {k: v for k, v in sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)}
    return list(leaderboard.keys()), list(leaderboard.values())


def trim_leaderboard(peptides, spectrum, n):
    leaderboard, scores = generate_leaderboard(peptides, spectrum)
    for i in range(n, len(leaderboard)):
        if scores[i] < scores[n-1]:
            for j in range(i, len(leaderboard)):
                leaderboard.pop()
            return leaderboard

    return leaderboard


def leaderboard_cyclopeptide_sequencing(spectrum, n):
    leaderboard = {""}
    leader = ""
    while len(leaderboard) > 0:
        leaderboard = expand(spectrum, leaderboard, n, True)
        for candidate in leaderboard:
            if mass(candidate) == parent_mass(spectrum):
                if peptide_scoring(candidate, spectrum) > peptide_scoring(leader, spectrum):
                    leader = candidate
    return leader


def matching_score_candidates(peptide, spectrum):
    # TODO: This algorithm is on development
    max_score = peptide_scoring(peptide, spectrum, True)
    candidates = {""}
    matches = set()
    while len(candidates) > 0:
        candidates = expand(spectrum, candidates, with_leaderboard=True)
        for candidate in candidates:
            if mass(candidate) == parent_mass(spectrum):
                if peptide_scoring(candidate, spectrum) == max_score:
                    matches.add(candidate)

    return matches


def spectral_convolution(spectrum):
    spectral_aa = list()
    for aa_mass_1 in spectrum:
        for aa_mass_2 in spectrum:
            diff = aa_mass_2 - aa_mass_1
            if diff > 0:
                spectral_aa.append(diff)
    return spectral_aa


def most_frequent_convolution_aa(m, spectrum):
    convolved_spectrum = sorted([x for x in spectral_convolution(spectrum) if 57 <= x <= 200])
    convolved_counter = Counter(convolved_spectrum)
    freq_vector = sorted(convolved_spectrum, key=lambda x: -convolved_counter[x])
    for i in range(m, len(freq_vector)):
        if freq_vector[i - 1] != freq_vector[i]:
            for j in range(i, len(freq_vector)):
                freq_vector.pop()
            return freq_vector
    return freq_vector


def convolution_cyclopeptide_sequencing(spectrum, m, n):
    # TODO: This algorithm is under development
    extended_spectrum = most_frequent_convolution_aa(m, spectrum)
    extended_spectrum.extend(spectrum)
    leaderboard = [[]]
    leader = ""
    while len(leaderboard) > 0:
        extended = list()
        for aa_mass in leaderboard:
            for i in range(57, 201):
                aa_mass.append(i)
                extended.append(aa_mass)
        leaderboard = extended

    return leader


def main():
    np.random.seed(2018)

    if parser.input is not None:
        try:
            f = open(parser.input, 'r')
            args = [x for x in f.read().split('\n') if x != ""]
        finally:
            f.close()
    else:
        args = cmd_input()

    if parser.kmer_composition:
        print_list(kmer_composition(int(args[0]), args[1]))
    elif parser.path_to_genome:
        print(path_to_genome(args))
    elif parser.overlap_graph:
        graph, graph_list = overlap_graph(sorted(args))
        print_graph(graph_list)
    elif parser.de_Bruijn_graph:
        graph, graph_list = de_bruijn_graph(int(args[0]), args[1])
        print_graph(graph_list)
    elif parser.de_Bruijn_graph_kmers:
        graph = de_bruijn_graph_kmers(args)
        print_graph(graph)
    elif parser.Eulerian_cycle:
        graph = input_graph(args)
        path = eulerian_cycle(graph)
        print_path(path)
    elif parser.Eulerian_path:
        graph = input_graph(args)
        path = eulerian_path(graph)
        print_path(path)
    elif parser.genome_reconstruction:
        genome = string_reconstruction(int(args[0]), args[1:])
        print_text(genome)
    elif parser.k_universal_circular_string:
        text = k_universal_circular_string(int(args[0]))
        print_text(text)
    elif parser.string_spelled_gapped_pattern:
        args1 = args[0].split()
        pairs = [(pair.split("|")[0], pair.split("|")[1]) for pair in args[1:]]
        genome = string_spelled_gapped_pattern(int(args1[0]), int(args1[1]), pairs)
        print_text(genome)
    elif parser.reconstruct_read_pairs:
        args1 = args[0].split()
        pairs = [(pair.split("|")[0], pair.split("|")[1]) for pair in args[1:]]
        genome = reconstruct_read_pairs(int(args1[0]), int(args1[1]), pairs)
        print_text(genome)
    elif parser.maximal_non_branching_path:
        graph = input_graph(args)
        paths = maximal_non_branching_paths(graph)
        for path in paths:
            print_path(path)
    elif parser.contig_generator:
        graph = de_bruijn_graph_kmers(args)
        paths = maximal_non_branching_paths(graph)
        contigs = ""
        for path in paths:
            contigs += string_reconstruction(len(path[0]), path) + " "
        print_text(contigs[:-1])
    elif parser.translate_RNA:
        protein = translate_rna(args[0])
        print_text(protein)
    elif parser.peptide_encoding:
        dna_sequences = peptide_encoding(args[0], args[1])
        print_list(dna_sequences)
    elif parser.number_of_dna_strings_translate_peptide:
        count = number_of_dna_strings_translate_peptide(args[0])
        print(count)
    elif parser.number_of_subpeptides_from_cyclic_peptide:
        result = str(number_of_subpeptides_from_peptide(int(args[0]), True))
        print_text(result)
    elif parser.number_of_subpeptides_from_linear_peptide:
        result = str(number_of_subpeptides_from_peptide(int(args[0])))
        print_text(result)
    elif parser.linear_theoretical_spectrum:
        ts = theoretical_spectrum(args[0])
        print_list(ts, delimiter=" ")
    elif parser.cyclic_theoretical_spectrum:
        ts = theoretical_spectrum(args[0], True)
        print_list(ts, delimiter=" ")
    elif parser.brute_force_cyclopeptide_sequencing:
        mass_spectrum = input_list(args[0], int)
        peptide = brute_force_cyclopeptide_sequencing(mass_spectrum)
        print_list(peptide, delimiter=" ")
    elif parser.cyclopeptide_sequencing:
        mass_spectrum = input_list(args[0], int)
        peptide = cyclopeptide_sequencing(mass_spectrum)
        mass_sequences = peptides_to_mass_sequence(peptide)
        print_list(mass_sequences, delimiter=" ")
    elif parser.cyclopeptide_scoring:
        mass_spectrum = input_list(args[1], int)
        score = peptide_scoring(args[0], mass_spectrum, True)
        print_text(score)
    elif parser.linear_peptide_scoring:
        mass_spectrum = input_list(args[1], int)
        score = peptide_scoring(args[0], mass_spectrum)
        print_text(score)
    elif parser.trim_leaderboard:
        peptide = input_list(args[0], str)
        spectrum = input_list(args[1], int)
        n = int(args[2])
        trimmed_lb = trim_leaderboard(peptide, spectrum, n)
        print_list(trimmed_lb, delimiter=" ")
    elif parser.leaderboard_cyclopeptide_sequencing:
        n = int(args[0])
        spectrum = input_list(args[1], int)
        peptide = leaderboard_cyclopeptide_sequencing(spectrum, n)
        mass_sequences = peptides_to_mass_sequence([peptide])
        print_list(mass_sequences, delimiter=" ")
    elif parser.find_maximum_score_leaderboard_cyclopeptide_sequencing:
        n = int(args[0])
        spectrum = input_list(args[1], int)
        leader = leaderboard_cyclopeptide_sequencing(spectrum, n)
        matching_peptides = matching_score_candidates(leader, spectrum)
        mass_sequences = peptides_to_mass_sequence(matching_peptides)
        print_list(mass_sequences, delimiter=" ")
    elif parser.spectral_convolution:
        spectrum = input_list(args[0], int)
        amino_acids = spectral_convolution(spectrum)
        print_list(amino_acids, delimiter=" ")
    elif parser.convolution_cyclopeptide_sequencing:
        m = int(args[0])
        n = int(args[1])
        spectrum = input_list(args[2], int)
        peptide = convolution_cyclopeptide_sequencing(spectrum, m, n)
        mass_sequences = peptides_to_mass_sequence([peptide])
        print_list(mass_sequences, delimiter=" ")


if __name__ == '__main__':
    parser = arg_setup()

    if parser.integer_mass_table is not None:
        imt = input_integer_mass_table(parser.integer_mass_table)

    main()