from nltk import PCFG, Tree

from nltk import nonterminals, Nonterminal, Production

import itertools

import sys

import random

import csv

import json

import gzip

from tqdm import tqdm

from typing import *


def generate(grammar, start=None, depth=None):

    """

    Generates an iterator of all sentences from a CFG.



    :param grammar: The Grammar used to generate sentences.

    :param start: The Nonterminal from which to start generate sentences.

    :param depth: The maximal depth of the generated tree.

    :param n: The maximum number of sentences to return.

    :return: An iterator of lists of terminal tokens.

    """

    if not start:

        start = grammar.start()

    if depth is None:

        depth = sys.maxsize



    items = [start]

    tree = _generate(grammar,items, depth)

    return tree[0]



def _generate(grammar,items,depth=None):

    if depth > 0:

        result = []

        for i in items:

            p = random.random()

            total_rule_prob = 0.

            if isinstance(i, Nonterminal):

                for prod in grammar.productions(lhs=i):

                    total_rule_prob += prod.prob()

                    if p < total_rule_prob:

                        expansion = _generate(grammar, prod.rhs(), depth - 1)

                        result += [Tree(i, expansion)]

                        break

            else:

                result += [i]  

                break              

        return result


S, S2, NP, MP, AdvP, VPTr, RelP, NPTr, VP, DetC, DetN, DNPc, DNPn, Nc, Nn, PN, Pron, M, VInTr, VTr, NTrc, PlDet, PlNTr, RP, V, NTand, Adv, comma = nonterminals('S, S2, NP, MP, AdvP, VPTr, RelP, NPTr, VP, DetC, DetN, DNPc, DNPn, Nc, Nn, PN, Pron, M, VInTr, VTr, NTrc, PlDet,PlNTr, RP, V, NTand, Adv, comma')



neg_grammar = PCFG.fromstring("""

    S -> NP MP [0.33] | NP M VPTr RelP [0.33] | AdvP S2 [0.34]

    S2 -> M NP VInTr [0.33] | M NP VPTr RelP [0.33] | NTand AdvP S2 [0.34]

    NP -> DNPc [0.1] | DNPn [0.1] | PN [0.6] | Pron [0.2]

    DNPc -> Nc DetC[1.0]

    DNPn -> Nn DetN[1.0]

    MP -> M VInTr [0.5] | M VPTr [0.5]

    AdvP -> Adv NP MP comma [1.0]

    VPTr -> VTr NPTrc  [1.0]

    RelP -> RP VP [1.0]

    NPTrc -> NTrc DetC [1.0]

    VP -> NP V [1.0]

    DetC -> '-en' [1.0]

    DetN -> '-et' [1.0]

    Nc -> 'student' [0.3] | 'katt' [0.3] | 'hund' [0.2] | 'frisör' [0.2]

    Nn -> 'svin' [0.3] | 'lag' [0.3] | 'djur' [0.4]

    PN -> 'Harry' [0.1] | 'Hermione' [0.1] | 'Ron' [0.1] | 'Petunia' [0.05] | 'Vernon' [0.05] | 'Lily' [0.1] | 'James' [0.1] | 'Snape' [0.1] | 'McGonagall' [0.1] | 'Draco' [0.05] | 'Tom' [0.05] | 'Albus' [0.1] 

    Pron -> 'han' [0.5] | 'hon' [0.5]

    M -> 'kan' [0.2] | 'måste' [0.2] | 'får' [0.3] | 'ska' [0.3] 

    VInTr -> 'dö' [0.2] | 'le'[0.2] | 'hoppa' [0.1] | 'skratta' [0.1] | 'festa' [0.1] | 'spy' [0.1] | 'nysa' [0.1] | 'springa' [0.1]

    VTr -> 'äta' [0.4] | 'baka' [0.3] | 'krossa' [0.2] | 'smaka' [0.1]

    NTrc -> 'korv' [0.2] | 'paj' [0.2] | 'lax' [0.2] | 'julmust' [0.2] | 'smörgås' [0.2]

    RP -> 'som' [1.0]

    V -> 'älskar' [0.2] | 'ser' [0.2] | 'hatar' [0.2] | 'gillar' [0.2] | 'ogillar' [0.2]

    NTand -> 'och' [1.0]

    Adv -> 'eftersom' [0.5] | 'trots att' [0.5]

    comma -> ',' [1.0]   

""")




 
def create_file (filename, grammar, ex_generator, n=10):

    with open("test_file.csv", mode='w') as output_file:

        output_writer = csv.writer(output_file, delimiter=',', quotechar=' ')

#        output_writer.writerow({'SRC', 'TRANSFORM', 'TRG'})

        output_writer.writerow(['SRC', 'TRG'])

        for _ in range(n):

            # src, trans, targ = ex_generator(grammar)

            (pos), src, (neg), targ = ex_generator(grammar)

            # output_writer.writerow([src + ' ' + trans, targ])

            print([(pos) + src + ' ' + (neg) + targ])






# 3 cases: t[1] can be an M, MP, or an S2

# to form the tree, I use recursion. The base case is M or MP and the recursive call happens in the case of S2

def negate(t):

    symbol = t[1].label().symbol()
    if t[0].label().symbol() == 'M':
      if t[2].label().symbol() == 'VInTr':
        v = t[2,0]
        v = 'inte ' + v
        t[2,0] = v
      else:
        v = t[2,0,0]
        v = 'inte ' + v
        t[2,0,0] = v
      return t

    # base case 1

    elif symbol == 'M':

        modal = t[1,0]

        modal = modal + ' inte'

        t[1,0] = modal

    # base case 2

    elif symbol == 'MP':

        modal = t[1,0]

        modal = modal[-1]

        modal = modal + ' inte'

        t[1,0] = modal

    elif symbol == 'NP':
      modal = t[0,0]
      modal = modal + ' inte'
      t[0,0] = modal

    # recursive call

    else:
        if t[1].label().symbol() == 'AdvP':
            negate(t[2])

        else:

            negate(t[1])  

    return t

def affirmation(grammar):
    pos_tree = generate(grammar)

    pos = ' '.join(pos_tree.leaves()).replace("- ","").replace(" ,",",")

    source = pos

    target = pos

    (pos) = 'pos'

    return source, (pos), target

def negation(grammar):

    pos_tree = generate(grammar)

    pos = ' '.join(pos_tree.leaves()).replace(" -","").replace(" ,",",")

    neg_tree = negate(pos_tree)

    neg = ' '.join(neg_tree.leaves()).replace(" -","").replace(" ,",",")

    source = pos

    target = neg

    (pos) = 'pos'

    (neg) = 'neg'

    return source, (neg), target

def neg_or_pos(grammar,p=0.5):
  if random.random() < p:
    return affirmation(grammar)
  else:
    return negation(grammar)