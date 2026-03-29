import itertools as it
import os
import sys
import time

import numpy as np
from numpy import einsum

np.set_printoptions(linewidth=300, precision=5)

from sympy import *


def group(iterable, n=2):
    return zip(*([iter(iterable)] * n))


def set_partition(iterable, n=2):
    set_partitions = set()

    for permutation in it.permutations(iterable):
        grouped = group(list(permutation), n)
        sorted_group = tuple(sorted([tuple(sorted(partition)) for partition in grouped]))
        set_partitions.add(sorted_group)

    return set_partitions


fourth_partitions = set_partition(["i", "j", "k", "l"], 2)
six_partitions = set_partition(["i", "j", "k", "l", "m", "n"], 2)
eight_partitions = set_partition(["i", "j", "k", "l", "m", "n", "o", "p"], 2)
fourth_partition_strings = []
six_partition_strings = []
eight_partition_strings = []
for i, partition in enumerate(fourth_partitions):
    fourth_partition_strings.append(
        "%s%s%s%s" % (partition[0][0], partition[0][1], partition[1][0], partition[1][1])
    )
for i, partition in enumerate(six_partitions):
    six_partition_strings.append(
        "%s%s%s%s%s%s"
        % (
            partition[0][0],
            partition[0][1],
            partition[1][0],
            partition[1][1],
            partition[2][0],
            partition[2][1],
        )
    )
for i, partition in enumerate(eight_partitions):
    eight_partition_strings.append(
        "%s%s%s%s%s%s%s%s"
        % (
            partition[0][0],
            partition[0][1],
            partition[1][0],
            partition[1][1],
            partition[2][0],
            partition[2][1],
            partition[3][0],
            partition[3][1],
        )
    )
eight_partition_strings = [
    "ijklmnop",
    "iljmkpno",
    "ikjolmnp",
    "iljokmnp",
    "iljkmonp",
    "imjnkplo",
    "iljnkpmo",
    "ipjlkmno",
    "iojmknlp",
    "iljnkmop",
    "ijknlomp",
    "imjnkolp",
    "ipjnkolm",
    "ijklmpno",
    "ikjolpmn",
    "ikjplmno",
    "ipjlknmo",
    "imjlkpno",
    "iljmknop",
    "ipjklmno",
    "injpkmlo",
    "ipjokmln",
    "injpklmo",
    "ikjplomn",
    "iojlknmp",
    "ipjmknlo",
    "injlkomp",
    "injokplm",
    "iojnkplm",
    "ikjlmpno",
    "iojklnmp",
    "ikjplnmo",
    "ijkolpmn",
    "iojlkmnp",
    "imjokpln",
    "imjlknop",
    "ipjlkomn",
    "iljmkonp",
    "ipjklnmo",
    "imjpkoln",
    "iljpkmno",
    "imjpklno",
    "ijkplomn",
    "iojmkpln",
    "ijknlmop",
    "injlkmop",
    "ijkmlpno",
    "injoklmp",
    "imjlkonp",
    "imjpknlo",
    "ijknlpmo",
    "ipjklomn",
    "ijkmlonp",
    "injmkplo",
    "imjklpno",
    "ijklmonp",
    "iljokpmn",
    "ikjolnmp",
    "injmkolp",
    "imjklonp",
    "ijkmlnop",
    "injmklop",
    "iojpklmn",
    "injklomp",
    "ikjmlpno",
    "imjklnop",
    "injpkolm",
    "iljnkomp",
    "imjoklnp",
    "injlkpmo",
    "ikjlmnop",
    "ikjnlomp",
    "ikjlmonp",
    "ipjnkmlo",
    "iljpknmo",
    "injklpmo",
    "ipjnklmo",
    "imjoknlp",
    "iojnklmp",
    "ikjmlnop",
    "ijkolnmp",
    "iljkmnop",
    "iojpknlm",
    "ipjoklmn",
    "ikjnlpmo",
    "iojlkpmn",
    "iojklmnp",
    "imjnklop",
    "iljkmpno",
    "iojklpmn",
    "ipjmkoln",
    "ipjmklno",
    "injokmlp",
    "iljpkomn",
    "iojnkmlp",
    "ikjmlonp",
    "injklmop",
    "iljoknmp",
    "ijkplnmo",
    "iojmklnp",
    "ipjoknlm",
    "ikjnlmop",
    "ijkplmno",
    "iojpkmln",
    "ijkolmnp",
]

stringy = "ijklmnop"


def make_integrals(matrix, order=2):
    input_matrix = 0.5 * matrix  # This is just the covariance matrix
    fourth_order_sum_elements = einsum("ij,kl->ijkl", input_matrix, input_matrix)
    if order == 4:
        fourth_order_integrals = summaker(fourth_order_sum_elements, fourth_partition_strings)
        return fourth_order_integrals
    sixth_order_sum_elements = einsum(
        "ijkl,mn->ijklmn", fourth_order_sum_elements, input_matrix
    )
    if order == 6:
        sixth_order_integrals = summaker(sixth_order_sum_elements, six_partition_strings)
        return sixth_order_integrals
    eight_order_sum_elements = einsum(
        "ijklmn,op->ijklmnop", sixth_order_sum_elements, input_matrix
    )
    if order == 8:
        return summaker(eight_order_sum_elements, eight_partition_strings)
    return sixth_order_integrals


def summaker(sum_elements, partition_strings):
    expectations = np.zeros_like(sum_elements)
    length = len(sum_elements.shape)

    for i in range(len(partition_strings)):
        permutation = partition_strings[i]
        expectations += einsum("%s->%s" % (permutation, stringy[:length]), sum_elements)
    return expectations


strings_4 = [
    "ij,kl->ijkl",
    "kl,ij->ijkl",
    "ik,jl->ijkl",
    "il,jk->ijkl",
    "jk,il->ijkl",
    "jl,ik->ijkl",
]
strings5_3 = [
    "klm,ij->ijklm",
    "ijk,lm->ijklm",
    "ijl,km->ijklm",
    "ijm,kl->ijklm",
    "ikl,jm->ijklm",
    "ikm,jl->ijklm",
    "ilm,jk->ijklm",
    "jkl,im->ijklm",
    "jkm,il->ijklm",
    "jlm,ik->ijklm",
]
strings5_1 = [
    "i,jklm->ijklm",
    "j,iklm->ijklm",
    "k,ijlm->ijklm",
    "l,ijkm->ijklm",
    "m,ijkl->ijklm",
]
strings6_2 = [
    "ij,klmn->ijklmn",
    "ik,jlmn->ijklmn",
    "il,jkmn->ijklmn",
    "im,jkln->ijklmn",
    "in,jklm->ijklmn",
    "jk,ilmn->ijklmn",
    "jl,ikmn->ijklmn",
    "jm,ikln->ijklmn",
    "jn,iklm->ijklmn",
    "kl,ijmn->ijklmn",
    "km,ijln->ijklmn",
    "kn,ijlm->ijklmn",
    "lm,ijkn->ijklmn",
    "ln,ijkm->ijklmn",
    "mn,ijkl->ijklmn",
]
strings6_4 = [
    "ijkl,mn->ijklmn",
    "ijkm,ln->ijklmn",
    "ijkn,lm->ijklmn",
    "ijlm,kn->ijklmn",
    "ijln,km->ijklmn",
    "ijmn,kl->ijklmn",
    "iklm,jn->ijklmn",
    "ikln,jm->ijklmn",
    "ikmn,jl->ijklmn",
    "ilmn,jk->ijklmn",
    "jklm,in->ijklmn",
    "jkln,im->ijklmn",
    "jkmn,il->ijklmn",
    "jlmn,ik->ijklmn",
    "klmn,ij->ijklmn",
]
strings7_5 = [
    "ijklm,no->ijklmno",  # ijk
    "ijkln,mo->ijklmno",
    "ijklo,mn->ijklmno",
    "ijkmn,lo->ijklmno",
    "ijkmo,ln->ijklmno",
    "ijkno,lm->ijklmno",
    "ijlmn,ko->ijklmno",  # ij
    "ijlmo,kn->ijklmno",
    "ijlno,km->ijklmno",
    "iklmn,jo->ijklmno",  # kl
    "iklmo,jn->ijklmno",
    "iklno,jm->ijklmno",
    "jklmn,io->ijklmno",
    "jklmo,in->ijklmno",
    "jklno,im->ijklmno",
    "klmno,ij->ijklmno",
    "ijmno,kl->ijklmno",  # mno
    "ikmno,jl->ijklmno",
    "ilmno,jk->ijklmno",
    "jkmno,il->ijklmno",
    "jlmno,ik->ijklmno",
]


strings7_3 = [
    "ijk,lmno->ijklmno",
    "ijl,kmno->ijklmno",
    "ijm,klno->ijklmno",
    "ijn,klmo->ijklmno",
    "ijo,klmn->ijklmno",
    "ikl,jmno->ijklmno",
    "ikm,jlno->ijklmno",
    "ikn,jlmo->ijklmno",
    "iko,jlmn->ijklmno",
    "ilm,jkno->ijklmno",
    "iln,jkmo->ijklmno",
    "ilo,jkmn->ijklmno",
    "imn,jklo->ijklmno",
    "imo,jkln->ijklmno",
    "ino,jklm->ijklmno",
    "jkl,imno->ijklmno",
    "jkm,ilno->ijklmno",
    "jkn,ilmo->ijklmno",
    "jko,ilmn->ijklmno",
    "jlm,ikno->ijklmno",
    "jln,ikmo->ijklmno",
    "jlo,ikmn->ijklmno",
    "jmn,iklo->ijklmno",
    "jmo,ikln->ijklmno",
    "jno,iklm->ijklmno",
    "klm,ijno->ijklmno",
    "kln,ijmo->ijklmno",
    "klo,ijmn->ijklmno",
    "kmn,ijlo->ijklmno",
    "kmo,ijln->ijklmno",
    "kno,ijlm->ijklmno",
    "lmn,ijko->ijklmno",
    "lmo,ijkn->ijklmno",
    "lno,ijkm->ijklmno",
    "mno,ijkl->ijklmno",
]
strings7_1 = [
    "i,jklmno->ijklmno",
    "j,iklmno->ijklmno",
    "k,ijlmno->ijklmno",
    "l,ijkmno->ijklmno",
    "m,ijklno->ijklmno",
    "n,ijklmo->ijklmno",
    "o,ijklmn->ijklmno",
]

strings8_6 = [
    "ijklmn,op->ijklmnop",  # ij contractions
    "ijklmo,np->ijklmnop",
    "ijklmp,no->ijklmnop",
    "ijklno,mp->ijklmnop",
    "ijklnp,mo->ijklmnop",
    "ijklop,mn->ijklmnop",
    "ijkmno,lp->ijklmnop",
    "ijkmnp,lo->ijklmnop",
    "ijkmop,ln->ijklmnop",
    "ijknop,lm->ijklmnop",
    "ijlnom,kp->ijklmnop",
    "ijlmnp,ko->ijklmnop",
    "ijlmop,kn->ijklmnop",
    "ijlnop,km->ijklmnop",
    "ijmnop,kl->ijklmnop",
    "iklmop,jn->ijklmnop",  # op contractions
    "iklnop,jm->ijklmnop",
    "ikmnop,jl->ijklmnop",
    "ilmnop,jk->ijklmnop",
    "jklmop,in->ijklmnop",
    "jklnop,im->ijklmnop",
    "jkmnop,il->ijklmnop",
    "jlmnop,ik->ijklmnop",
    "klmnop,ij->ijklmnop",  # klm contractions
    "jklmno,ip->ijklmnop",
    "jklmnp,io->ijklmnop",
    "iklmno,jp->ijklmnop",
    "iklmnp,jo->ijklmnop",
]
strings8_4 = [
    "ijkl,mnop->ijklmnop",  # ij contraction
    "ijkm,lnop->ijklmnop",
    "ijkn,lmop->ijklmnop",
    "ijko,lmnp->ijklmnop",
    "ijkp,lmno->ijklmnop",
    "ijlm,knop->ijklmnop",
    "ijln,kmop->ijklmnop",
    "ijlo,kmnp->ijklmnop",
    "ijlp,kmno->ijklmnop",
    "ijmn,klop->ijklmnop",
    "ijmo,klnp->ijklmnop",
    "ijmp,klno->ijklmnop",
    "ijno,klmp->ijklmnop",
    "ijnp,klmo->ijklmnop",
    "ijop,klmn->ijklmnop",
    "iklm,jnop->ijklmnop",  # ik contraction
    "ikln,jmop->ijklmnop",
    "iklo,jmnp->ijklmnop",
    "iklp,jmno->ijklmnop",
    "ikmn,jlop->ijklmnop",
    "ikmo,jlnp->ijklmnop",
    "ikmp,jlno->ijklmnop",
    "ikno,jlmp->ijklmnop",
    "iknp,jlmo->ijklmnop",
    "ikop,jlmn->ijklmnop",  # OP contraction
    "jlop,ikmn->ijklmnop",
    "ilop,jkmn->ijklmnop",
    "mnop,ijkl->ijklmnop",
    "imop,jkln->ijklmnop",
    "inop,jklm->ijklmnop",
    "klop,ijmn->ijklmnop",
    "jkop,ilmn->ijklmnop",
    "lmop,ijkn->ijklmnop",
    "lnop,ijkm->ijklmnop",
    "knop,ijlm->ijklmnop",
    "jmop,ikln->ijklmnop",
    "jnop,iklm->ijklmnop",
    "lmkn,ijop->ijklmnop",  # ml contraction
    "lmko,ijnp->ijklmnop",
    "lmkp,ijno->ijklmnop",
    "lmno,ijkp->ijklmnop",
    "lmnp,ijko->ijklmnop",
    "lmjk,inop->ijklmnop",
    "lmin,jkop->ijklmnop",
    "lmio,jknp->ijklmnop",
    "lmip,jkno->ijklmnop",
    "lmjn,ikop->ijklmnop",
    "lmjo,iknp->ijklmnop",
    "lmjp,ikno->ijklmnop",
    "jlno,ikmp->ijklmnop",
    "jlnp,ikmo->ijklmnop",
    "imnp,jklo->ijklmnop",
    "mjno,iklp->ijklmnop",
    "klno,ijmp->ijklmnop",
    "klnp,ijmo->ijklmnop",
    "jkln,imop->ijklmnop",
    "jklo,imnp->ijklmnop",
    "jklp,imno->ijklmnop",
    "jkmn,ilop->ijklmnop",
    "jkmo,ilnp->ijklmnop",
    "jkmp,ilno->ijklmnop",
    "jkno,ilmp->ijklmnop",
    "jknp,ilmo->ijklmnop",
    "ilno,jkmp->ijklmnop",
    "ilnp,jkmo->ijklmnop",
    "imno,jklp->ijklmnop",
    "kmno,ijlp->ijklmnop",
    "kmnp,ijlo->ijklmnop",
    "kmop,ijln->ijklmnop",
    "mjnp,iklo->ijklmnop",
]

strings8_2 = [
    "ij,klmnop->ijklmnop",
    "ik,jlmnop->ijklmnop",
    "il,jkmnop->ijklmnop",
    "im,jklnop->ijklmnop",
    "in,jklmop->ijklmnop",
    "io,jklmnp->ijklmnop",
    "ip,jklmno->ijklmnop",
    "jk,ilmnop->ijklmnop",
    "jl,ikmnop->ijklmnop",
    "jm,iklnop->ijklmnop",
    "jn,iklmop->ijklmnop",
    "jo,iklmnp->ijklmnop",
    "jp,iklmno->ijklmnop",
    "kl,ijmnop->ijklmnop",
    "km,ijlnop->ijklmnop",
    "kn,ijlmop->ijklmnop",
    "ko,ijlmnp->ijklmnop",
    "kp,ijlmno->ijklmnop",
    "lm,ijknop->ijklmnop",
    "ln,ijkmop->ijklmnop",
    "lo,ijkmnp->ijklmnop",
    "lp,ijkmno->ijklmnop",
    "mn,ijklop->ijklmnop",
    "mo,ijklnp->ijklmnop",
    "mp,ijklno->ijklmnop",
    "no,ijklmp->ijklmnop",
    "np,ijklmo->ijklmnop",
    "op,ijklmn->ijklmnop",
]


# @numba.jit
def calculate_full_expectation_value(matrix, mu, order=2, taylor_order=8):
    inp = matrix / 2
    if order == 1:
        return mu
    mu2 = np.outer(mu, mu)
    if order == 2:
        return inp + mu2
    mu3 = einsum("i,jk->ijk", mu, mu2)
    if order == 3:
        return (
            mu3
            + einsum("i,jk->ijk", mu, inp)
            + einsum("j,ik->ijk", mu, inp)
            + einsum("k,ij->ijk", mu, inp)
        )
    if taylor_order >= 4:
        fourth_expecs = make_integrals(matrix, order=4)
    mu4 = einsum("i,jkl->ijkl", mu, mu3)
    if order == 4:
        returnval = mu4
        for stringy in strings_4:
            returnval += +einsum("%s" % stringy, mu2, inp)
        if taylor_order >= 4:
            returnval += fourth_expecs
        return returnval
    mu5 = einsum("i,jklm->ijklm", mu, mu4)
    if order == 5:
        returnval = mu5
        for stringy in strings5_3:
            returnval += +einsum("%s" % stringy, mu3, inp)

        if taylor_order >= 4:

            for stringy in strings5_1:
                returnval += +einsum("%s" % stringy, mu, fourth_expecs)
        return returnval

    mu6 = einsum("i,jklmn->ijklmn", mu, mu5)
    if taylor_order >= 6:
        sixth_expecs = make_integrals(matrix, order=6)
    if order == 6:
        returnval = mu6
        for stringy in strings6_4:
            returnval += +einsum("%s" % stringy, mu4, inp)

        if taylor_order >= 4:
            for stringy in strings6_2:
                returnval += +einsum("%s" % stringy, mu2, fourth_expecs)
        if taylor_order >= 6:
            returnval += sixth_expecs
        return returnval
    mu7 = einsum("i,jklmno->ijklmno", mu, mu6)
    if order == 7:
        returnval = mu7

        for stringy in strings7_5:
            returnval += einsum("%s" % stringy, mu5, inp)
        if taylor_order >= 4:
            for stringy in strings7_3:
                returnval += einsum("%s" % stringy, mu3, fourth_expecs)
        if taylor_order >= 6:
            for stringy in strings7_1:
                returnval += einsum("%s" % stringy, mu, sixth_expecs)
        return returnval
    mu8 = einsum("i,jklmnop->ijklmnop", mu, mu7)
    if taylor_order >= 8:
        eigth_expecs = make_integrals(matrix, order=8)
    if order == 8:
        returnval = mu8
        for stringy in strings8_6:
            returnval += einsum("%s" % stringy, mu6, inp)
        if taylor_order >= 4:
            for stringy in strings8_4:
                returnval += einsum("%s" % stringy, mu4, fourth_expecs)
        if taylor_order >= 6:
            for stringy in strings8_2:
                returnval += einsum("%s" % stringy, mu2, sixth_expecs)
        if taylor_order >= 8:
            returnval += eigth_expecs
        return returnval


def get_x7_x8_contraction(matrix, mu):
    inp = matrix / 2
    mu3x = einsum("i,i,i->i", mu[1:], mu[1:], mu[1:])
    fourth_expecs = make_integrals(matrix, order=4)
    sixth_expecs = make_integrals(matrix, order=6)
    contraction_1 = einsum("k,l,i,j->kl", mu, mu, mu3x, mu3x)
    contraction_1 += 6 * einsum("i,j,k,l,ll->ij", mu, mu, mu3x, mu[1:], inp[1:, 1:])
    contraction_1 += 9 * einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[1:], mu[1:], mu[1:], mu[1:], inp[1:, 1:]
    )
    temp = einsum("i,l,k,l,jl->ij", mu, mu[1:], mu3x, mu[1:], inp[:, 1:])
    contraction_1 += 6 * temp
    contraction_1 += 6 * temp.T
    contraction_1 += einsum("k,l,ij->ij", mu3x, mu3x, inp[:, :])
    contraction_1 += 3 * einsum(
        "i,j,k,k,klll->ij", mu, mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, 1:, 1:]
    )
    contraction_1 += 9 * einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, 1:, 1:]
    )
    contraction_1 += 3 * einsum(
        "i,j,l,l,kkkl->ij", mu, mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, 1:, 1:]
    )
    contraction_1 += 1 * einsum("i,k,jlll->ij", mu, mu3x, fourth_expecs[:, 1:, 1:, 1:])
    temp = 6 * einsum(
        "i,k,k,l,jkll->ij", mu, mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    contraction_1 += temp + temp.T
    temp = einsum("i,k,l,l,jkkl->ij", mu, mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, 1:, 1:])
    contraction_1 += 9 * temp + 7 * temp.T
    contraction_1 += 3 * np.einsum(
        "k,l,ijkk->ij",
        mu[1:],
        mu3x,
        fourth_expecs[
            :,
            :,
            1:,
            1:,
        ],
    )
    temp = np.einsum(
        "i,l,jkkk->ij",
        mu,
        mu3x,
        fourth_expecs[
            :,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += temp + temp.T
    contraction_1 += 9 * np.einsum(
        "k,k,l,l,ijkl->ij",
        mu[1:],
        mu[1:],
        mu[1:],
        mu[1:],
        fourth_expecs[
            :,
            :,
            1:,
            1:,
        ],
    )
    contraction_1 += 3 * np.einsum(
        "k,l,ijll->ij",
        mu3x,
        mu[1:],
        fourth_expecs[
            :,
            :,
            1:,
            1:,
        ],
    )
    contraction_1 += np.einsum(
        "k,k,j,k,illl->ij",
        mu[1:],
        mu[1:],
        mu,
        mu[1:],
        fourth_expecs[
            :,
            1:,
            1:,
            1:,
        ],
    )
    temp = 3 * np.einsum(
        "k,k,i,l,jkll->ij",
        mu[1:],
        mu[1:],
        mu,
        mu[1:],
        fourth_expecs[
            :,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += temp + temp.T
    contraction_1 += 2 * np.einsum(
        "k,j,l,l,ikkl->ij",
        mu[1:],
        mu,
        mu[1:],
        mu[1:],
        fourth_expecs[
            :,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += np.einsum(
        "i,j,kkklll->ij",
        mu,
        mu,
        sixth_expecs[
            1:,
            1:,
            1:,
            1:,
            1:,
            1:,
        ],
    )
    temp = 3 * np.einsum(
        "i,k,jkklll->ij",
        mu,
        mu[1:],
        sixth_expecs[
            :,
            1:,
            1:,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += temp + temp.T
    temp = 3 * np.einsum(
        "i,l,jkkkll->ij",
        mu,
        mu[1:],
        sixth_expecs[
            :,
            1:,
            1:,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += temp + temp.T
    temp = 3 * np.einsum(
        "k,k,ijklll->ij",
        mu[1:],
        mu[1:],
        sixth_expecs[
            :,
            :,
            1:,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += temp + temp.T
    contraction_1 += 9 * np.einsum(
        "k,l,ijkkll->ij",
        mu[1:],
        mu[1:],
        sixth_expecs[
            :,
            :,
            1:,
            1:,
            1:,
            1:,
        ],
    )
    contraction_1 += 9 * np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[1:, 1:], inp[1:, 1:], inp[1:, 1:]
    )
    contraction_1 += 36 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, 1:], inp[:, 1:], inp[1:, 1:], inp[1:, 1:]
    )
    contraction_1 += 18 * np.einsum(
        "ik,jl,kk,ll->ij", inp[:, 1:], inp[:, 1:], inp[1:, 1:], inp[1:, 1:]
    )
    contraction_1 += 36 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, 1:], inp[:, 1:], inp[1:, 1:], inp[1:, 1:]
    )
    contraction_1 += 6 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[1:, 1:], inp[1:, 1:], inp[1:, 1:]
    )

    """Next step: Calculate contractions 2 and 3..."""
    mu4 = einsum("i,j,k,l->ijkl", mu, mu, mu, mu)
    mu6 = einsum("i,j,klmn->ijklmn", mu, mu, mu4)
    mu3x_new = einsum("i,i,i->i", mu[:-1], mu[:-1], mu[1:])
    contraction_2 = einsum("k,l,i,j->kl", mu, mu, mu3x_new, mu3x_new)
    contraction_2 += 2 * np.einsum("i,j,k,l,ll->ij", mu, mu, mu3x_new, mu[:-1], inp[:-1, 1:])
    contraction_2 += np.einsum("i,j,k,l,ll->ij", mu, mu, mu3x_new, mu[1:], inp[:-1, :-1])
    contraction_2 += np.einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[:-1], mu[:-1], mu[:-1], mu[:-1], inp[1:, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[:-1], mu[:-1], mu[:-1], mu[1:], inp[1:, :-1]
    )
    contraction_2 += np.einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[:-1], mu[1:], mu[:-1], mu[:-1], inp[:-1, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[:-1], mu[1:], mu[:-1], mu[1:], inp[:-1, :-1]
    )
    contraction_2 += 2 * np.einsum("i,j,k,l,kk->ij", mu, mu, mu[:-1], mu3x_new, inp[:-1, 1:])
    contraction_2 += np.einsum(
        "i,j,k,l,l,k,kl->ij", mu, mu, mu[:-1], mu[:-1], mu[:-1], mu[1:], inp[:-1, 1:]
    )
    contraction_2 += np.einsum("i,j,k,l,kk->ij", mu, mu, mu[1:], mu3x_new, inp[:-1, :-1])
    temp = np.einsum("i,k,k,l,jk->ij", mu, mu[:-1], mu[:-1], mu3x_new, inp[:, 1:])
    contraction_2 += temp + temp.T
    temp = 2 * np.einsum("j,k,k,l,ik->ij", mu, mu[:-1], mu[1:], mu3x_new, inp[:, :-1])
    contraction_2 += temp + temp.T
    contraction_2 += np.einsum("k,l,ij->ij", mu3x_new, mu3x_new, inp[:, :])
    temp = 2 * np.einsum("j,k,l,l,il->ij", mu, mu3x_new, mu[:-1], mu[1:], inp[:, :-1])
    contraction_2 += temp + temp.T
    temp = np.einsum("i,k,l,l,jl->ij", mu, mu3x_new, mu[:-1], mu[:-1], inp[:, 1:])
    contraction_2 += temp + temp.T
    contraction_2 += np.einsum(
        "i,j,k,k,klll->ij", mu, mu, mu[:-1], mu[:-1], fourth_expecs[1:, :-1, :-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "i,j,k,k,klll->ij", mu, mu, mu[:-1], mu[1:], fourth_expecs[:-1, :-1, :-1, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[:-1], mu[:-1], fourth_expecs[:-1, 1:, :-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[:-1], mu[1:], fourth_expecs[:-1, 1:, :-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[1:], mu[:-1], fourth_expecs[:-1, :-1, :-1, 1:]
    )
    contraction_2 += np.einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[1:], mu[1:], fourth_expecs[:-1, :-1, :-1, :-1]
    )
    contraction_2 += np.einsum(
        "i,j,l,l,kkkl->ij", mu, mu, mu[:-1], mu[:-1], fourth_expecs[:-1, :-1, 1:, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "i,j,l,l,kkkl->ij", mu, mu, mu[:-1], mu[1:], fourth_expecs[:-1, :-1, 1:, :-1]
    )
    temp = np.einsum("i,k,jlll->ij", mu, mu3x_new, fourth_expecs[:, :-1, :-1, 1:])
    contraction_2 += temp.T
    contraction_2 += 2 * temp
    temp = np.einsum(
        "j,k,k,l,ikll->ij", mu, mu[:-1], mu[1:], mu[1:], fourth_expecs[:, :-1, :-1, :-1]
    )
    contraction_2 += temp + temp.T
    temp = 4 * np.einsum(
        "j,k,l,l,ikkl->ij", mu, mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :-1, 1:, :-1]
    )
    contraction_2 += temp + temp.T
    contraction_2 += np.einsum("k,l,ijkk->ij", mu[1:], mu3x_new, fourth_expecs[:, :, :-1, :-1])
    contraction_2 += 2 * np.einsum(
        "k,k,l,l,ijkl->ij", mu[:-1], mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :, 1:, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "k,k,l,l,ijkl->ij", mu[:-1], mu[1:], mu[:-1], mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "k,l,ijkk->ij", mu[:-1], mu3x_new, fourth_expecs[:, :, :-1, 1:]
    )
    temp = np.einsum(
        "j,k,l,l,ikkl->ij", mu, mu[1:], mu[:-1], mu[1:], fourth_expecs[:, :-1, :-1, :-1]
    )
    contraction_2 += temp + 2 * temp.T
    contraction_2 += 2 * np.einsum(
        "k,l,ijll->ij", mu3x_new, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    contraction_2 += np.einsum("k,l,ijll->ij", mu3x_new, mu[1:], fourth_expecs[:, :, :-1, :-1])
    contraction_2 += 2 * np.einsum(
        "k,k,l,l,ijkl->ij", mu[:-1], mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    contraction_2 += np.einsum(
        "k,k,j,k,illl->ij", mu[:-1], mu[1:], mu, mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    temp = 2 * np.einsum(
        "k,k,i,l,jkll->ij", mu[:-1], mu[1:], mu, mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    contraction_2 += temp + temp.T
    temp = np.einsum(
        "k,k,i,l,jkll->ij", mu[:-1], mu[1:], mu, mu[1:], fourth_expecs[:, :-1, :-1, :-1]
    )
    contraction_2 += temp + temp.T
    temp = 2 * np.einsum(
        "j,k,l,l,ikkl->ij", mu, mu[:-1], mu[:-1], mu[:-1], fourth_expecs[:, :-1, 1:, 1:]
    )
    contraction_2 += temp + temp.T
    contraction_2 += np.einsum(
        "k,j,l,l,ikkl->ij", mu[1:], mu, mu[:-1], mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    contraction_2 += np.einsum(
        "k,k,l,l,ijkl->ij", mu[:-1], mu[:-1], mu[:-1], mu[:-1], fourth_expecs[:, :, 1:, 1:]
    )
    temp = np.einsum(
        "j,k,k,l,ikll->ij", mu, mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    contraction_2 += temp + temp.T
    temp = 2 * np.einsum(
        "i,k,k,l,jkll->ij", mu, mu[:-1], mu[1:], mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    contraction_2 += temp + temp.T
    contraction_2 += np.einsum(
        "i,k,l,l,jkkl->ij", mu, mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    contraction_2 += np.einsum(
        "k,j,l,l,ikkl->ij", mu[1:], mu, mu[:-1], mu[1:], fourth_expecs[:, :-1, :-1, :-1]
    )
    temp = 2 * np.einsum(
        "i,k,k,l,jkll->ij", mu, mu[:-1], mu[:-1], mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    contraction_2 += temp + temp.T

    contraction_2 += np.einsum(
        "i,j,kkklll->ij", mu, mu, sixth_expecs[:-1, :-1, 1:, :-1, :-1, 1:]
    )
    temp = 2 * np.einsum("i,k,jkklll->ij", mu, mu[:-1], sixth_expecs[:, :-1, 1:, :-1, :-1, 1:])
    contraction_2 += temp + temp.T
    temp = np.einsum("i,k,jkklll->ij", mu, mu[1:], sixth_expecs[:, :-1, :-1, :-1, :-1, 1:])
    contraction_2 += temp + temp.T

    temp = 2 * np.einsum("i,l,jkkkll->ij", mu, mu[:-1], sixth_expecs[:, :-1, :-1, 1:, :-1, 1:])
    contraction_2 += temp + temp.T
    temp = np.einsum("i,l,jkkkll->ij", mu, mu[1:], sixth_expecs[:, :-1, :-1, 1:, :-1, :-1])
    contraction_2 += temp + temp.T
    contraction_2 += np.einsum(
        "k,k,ijklll->ij", mu[:-1], mu[:-1], sixth_expecs[:, :, 1:, :-1, :-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "k,k,ijklll->ij", mu[:-1], mu[1:], sixth_expecs[:, :, :-1, :-1, :-1, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "k,l,ijkkll->ij", mu[:-1], mu[:-1], sixth_expecs[:, :, :-1, 1:, :-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "k,l,ijkkll->ij", mu[:-1], mu[1:], sixth_expecs[:, :, :-1, 1:, :-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "k,l,ijkkll->ij", mu[1:], mu[:-1], sixth_expecs[:, :, :-1, :-1, :-1, 1:]
    )
    contraction_2 += np.einsum(
        "k,l,ijkkll->ij", mu[1:], mu[1:], sixth_expecs[:, :, :-1, :-1, :-1, :-1]
    )
    contraction_2 += np.einsum(
        "l,l,ijkkkl->ij", mu[:-1], mu[:-1], sixth_expecs[:, :, :-1, :-1, 1:, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "l,l,ijkkkl->ij", mu[:-1], mu[1:], sixth_expecs[:, :, :-1, :-1, 1:, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[:-1, :-1], inp[1:, :-1], inp[:-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, :-1], inp[:, 1:], inp[:-1, 1:], inp[:-1, :-1]
    )
    temp = 4 * np.einsum(
        "il,jk,kk,ll->ij", inp[:, :-1], inp[:, :-1], inp[:-1, 1:], inp[:-1, 1:]
    )
    contraction_2 += temp + temp.T

    contraction_2 += 2 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, 1:], inp[:, :-1], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, :-1], inp[:, :-1], inp[:-1, 1:], inp[1:, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, :-1], inp[:, 1:], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[:-1, :-1], inp[:-1, :-1], inp[1:, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, 1:], inp[:, :-1], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, 1:], inp[:, :-1], inp[:-1, :-1], inp[1:, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, :-1], inp[:, 1:], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, 1:], inp[:, :-1], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, :-1], inp[:, 1:], inp[:-1, :-1], inp[1:, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, :-1], inp[:, 1:], inp[:-1, :-1], inp[1:, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, :-1], inp[:, :-1], inp[:-1, :-1], inp[1:, 1:]
    )
    temp = 2 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, 1:], inp[:, 1:], inp[:-1, :-1], inp[:-1, :-1]
    )
    contraction_2 += temp + temp.T
    contraction_2 += 2 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[:-1, :-1], inp[:-1, 1:], inp[1:, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[:-1, 1:], inp[:-1, :-1], inp[1:, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, :-1], inp[:, 1:], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, :-1], inp[:, :-1], inp[:-1, :-1], inp[1:, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, :-1], inp[:, :-1], inp[:-1, :-1], inp[1:, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, :-1], inp[:, :-1], inp[:-1, 1:], inp[1:, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, 1:], inp[:, :-1], inp[:-1, :-1], inp[1:, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, :-1], inp[:, :-1], inp[:-1, 1:], inp[:-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "ik,jl,kk,ll->ij", inp[:, 1:], inp[:, :-1], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "il,jk,kk,ll->ij", inp[:, :-1], inp[:, 1:], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[:-1, 1:], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[:-1, 1:], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, :-1], inp[:, :-1], inp[1:, 1:], inp[:-1, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, 1:], inp[:, :-1], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += np.einsum(
        "ik,jl,kk,ll->ij", inp[:, 1:], inp[:, 1:], inp[:-1, :-1], inp[:-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, 1:], inp[:, :-1], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 4 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, :-1], inp[:, 1:], inp[:-1, :-1], inp[:-1, 1:]
    )
    contraction_2 += 4 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, :-1], inp[:, :-1], inp[1:, :-1], inp[:-1, 1:]
    )
    contraction_2 += 2 * np.einsum(
        "il,jk,kk,ll->ij", inp[:, 1:], inp[:, :-1], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += 2 * np.einsum(
        "ik,jl,kk,ll->ij", inp[:, :-1], inp[:, 1:], inp[:-1, 1:], inp[:-1, :-1]
    )
    contraction_2 += np.einsum(
        "il,jk,kk,ll->ij", inp[:, 1:], inp[:, 1:], inp[:-1, :-1], inp[:-1, :-1]
    )
    contraction_2 += np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[:-1, :-1], inp[1:, 1:], inp[:-1, :-1]
    )

    contraction_3 = einsum("k,l,i,j->kl", mu, mu, mu3x, mu3x_new)
    contraction_3 += 2 * np.einsum(
        "i,j,k,k,k,l,ll->ij", mu, mu, mu[1:], mu[1:], mu[1:], mu[:-1], inp[:-1, 1:]
    )
    contraction_3 += np.einsum(
        "i,j,k,k,k,l,ll->ij", mu, mu, mu[1:], mu[1:], mu[1:], mu[1:], inp[:-1, :-1]
    )
    contraction_3 += 2 * np.einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[1:], mu[1:], mu[:-1], mu[:-1], inp[1:, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "i,j,k,k,l,l,kl->ij", mu, mu, mu[1:], mu[1:], mu[:-1], mu[1:], inp[1:, :-1]
    )
    contraction_3 += 3 * np.einsum("i,j,k,l,kk->ij", mu, mu, mu[1:], mu3x_new, inp[1:, 1:])
    contraction_3 += np.einsum(
        "i,j,k,l,l,k,kl->ij", mu, mu, mu[1:], mu[:-1], mu[:-1], mu[1:], inp[1:, 1:]
    )
    temp = 3 * np.einsum("i,k,k,l,jk->ij", mu, mu[1:], mu[1:], mu3x_new, inp[:, 1:])
    contraction_3 += temp + temp.T
    temp = 2 * np.einsum(
        "i,k,k,k,l,l,jl->ij", mu, mu[1:], mu[1:], mu[1:], mu[:-1], mu[1:], inp[:, :-1]
    )
    contraction_3 += temp + temp.T
    contraction_3 += np.einsum("k,k,k,l,ij->ij", mu[1:], mu[1:], mu[1:], mu3x_new, inp[:, :])
    temp = np.einsum(
        "j,k,k,k,l,l,il->ij", mu, mu[1:], mu[1:], mu[1:], mu[:-1], mu[:-1], inp[:, 1:]
    )
    contraction_3 += temp + temp.T

    contraction_3 += 3 * np.einsum(
        "i,j,k,k,klll->ij", mu, mu, mu[1:], mu[1:], fourth_expecs[1:, :-1, :-1, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[1:], mu[:-1], fourth_expecs[1:, 1:, :-1, 1:]
    )
    contraction_3 += 3 * np.einsum(
        "i,j,k,l,kkll->ij", mu, mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, :-1, :-1]
    )
    contraction_3 += np.einsum(
        "i,j,l,l,kkkl->ij", mu, mu, mu[:-1], mu[:-1], fourth_expecs[1:, 1:, 1:, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "i,j,l,l,kkkl->ij", mu, mu, mu[:-1], mu[1:], fourth_expecs[1:, 1:, 1:, :-1]
    )
    contraction_3 += np.einsum(
        "i,k,k,k,jlll->ij", mu, mu[1:], mu[1:], mu[1:], fourth_expecs[:, :-1, :-1, 1:]
    )
    contraction_3 += 4 * np.einsum(
        "i,k,k,l,jkll->ij", mu, mu[1:], mu[1:], mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "i,k,k,l,jkll->ij", mu, mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    contraction_3 += 3 * np.einsum(
        "i,k,l,l,jkkl->ij", mu, mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, 1:, 1:, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "i,k,l,l,jkkl->ij", mu, mu[1:], mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, :-1]
    )
    contraction_3 += 5 * np.einsum(
        "j,k,l,l,ikkl->ij", mu, mu[1:], mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, :-1]
    )
    contraction_3 += 3 * np.einsum(
        "k,l,l,l,ijkk->ij", mu[1:], mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    contraction_3 += np.einsum(
        "i,l,l,l,jkkk->ij", mu, mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "k,k,l,l,ijkl->ij", mu[1:], mu[1:], mu[:-1], mu[1:], fourth_expecs[:, :, 1:, :-1]
    )
    contraction_3 += np.einsum(
        "j,l,l,l,ikkk->ij", mu, mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "k,k,k,l,ijll->ij", mu[1:], mu[1:], mu[1:], mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    contraction_3 += np.einsum(
        "k,k,k,l,ijll->ij", mu[1:], mu[1:], mu[1:], mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    contraction_3 += 3 * np.einsum(
        "k,k,l,l,ijkl->ij", mu[1:], mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, :, 1:, 1:]
    )
    contraction_3 += np.einsum(
        "k,k,j,k,illl->ij", mu[1:], mu[1:], mu, mu[1:], fourth_expecs[:, :-1, :-1, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "k,k,i,l,jkll->ij", mu[1:], mu[1:], mu, mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    contraction_3 += np.einsum(
        "k,k,i,l,jkll->ij", mu[1:], mu[1:], mu, mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    contraction_3 += 2 * np.einsum(
        "k,k,j,l,ikll->ij", mu[1:], mu[1:], mu, mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    contraction_3 += np.einsum(
        "k,k,j,l,ikll->ij", mu[1:], mu[1:], mu, mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    contraction_3 += 2 * np.einsum(
        "j,k,l,l,ikkl->ij", mu, mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, 1:, 1:, 1:]
    )
    contraction_3 += np.einsum(
        "k,j,l,l,ikkl->ij", mu[1:], mu, mu[:-1], mu[:-1], fourth_expecs[:, 1:, 1:, 1:]
    )
    contraction_3 += 4 * np.einsum(
        "j,k,k,l,ikll->ij", mu, mu[1:], mu[1:], mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "j,k,k,l,ikll->ij", mu, mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    contraction_3 += np.einsum(
        "k,j,l,l,ikkl->ij", mu[1:], mu, mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, :-1]
    )

    contraction_3 += np.einsum(
        "i,j,kkklll->ij", mu, mu, sixth_expecs[1:, 1:, 1:, :-1, :-1, 1:]
    )
    temp = np.einsum("i,l,jkkkll->ij", mu, mu[1:], sixth_expecs[:, 1:, 1:, 1:, :-1, :-1])
    contraction_3 += temp + temp.T
    temp = 3 * np.einsum("j,k,ikklll->ij", mu, mu[1:], sixth_expecs[:, 1:, 1:, :-1, :-1, 1:])
    contraction_3 += temp + temp.T
    temp = 2 * np.einsum("i,l,jkkkll->ij", mu, mu[:-1], sixth_expecs[:, 1:, 1:, 1:, :-1, 1:])
    contraction_3 += temp + temp.T
    contraction_3 += 3 * np.einsum(
        "k,k,ijklll->ij", mu[1:], mu[1:], sixth_expecs[:, :, 1:, :-1, :-1, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "k,l,ijkkll->ij", mu[1:], mu[:-1], sixth_expecs[:, :, 1:, 1:, :-1, 1:]
    )
    contraction_3 += 3 * np.einsum(
        "k,l,ijkkll->ij", mu[1:], mu[1:], sixth_expecs[:, :, 1:, 1:, :-1, :-1]
    )
    contraction_3 += np.einsum(
        "l,l,ijkkkl->ij", mu[:-1], mu[:-1], sixth_expecs[:, :, 1:, 1:, 1:, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "l,l,ijkkkl->ij", mu[:-1], mu[1:], sixth_expecs[:, :, 1:, 1:, 1:, :-1]
    )

    contraction_3 += 6 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, 1:], inp[:, :-1], inp[1:, 1:], inp[1:, :-1]
    )
    contraction_3 += 6 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, :-1], inp[:, 1:], inp[1:, :-1], inp[1:, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[1:, :-1], inp[1:, :-1], inp[1:, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "ik,jl,kl,kl->ij", inp[:, 1:], inp[:, :-1], inp[1:, :-1], inp[1:, 1:]
    )
    temp = 6 * np.einsum("ik,jl,kl,kl->ij", inp[:, 1:], inp[:, 1:], inp[1:, :-1], inp[1:, :-1])
    contraction_3 += temp + temp.T
    contraction_3 += 6 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, 1:], inp[:, :-1], inp[1:, 1:], inp[1:, :-1]
    )
    contraction_3 += 6 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, :-1], inp[:, :-1], inp[1:, 1:], inp[1:, 1:]
    )
    contraction_3 += 2 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[1:, :-1], inp[1:, 1:], inp[1:, :-1]
    )
    contraction_3 += 2 * np.einsum(
        "ij,kl,kl,kl->ij", inp[:, :], inp[1:, 1:], inp[1:, :-1], inp[1:, :-1]
    )
    contraction_3 += 6 * np.einsum(
        "il,jk,kl,kl->ij", inp[:, :-1], inp[:, 1:], inp[1:, 1:], inp[1:, :-1]
    )
    contraction_3 += 6 * np.einsum(
        "il,jl,kk,kl->ij", inp[:, :-1], inp[:, 1:], inp[1:, 1:], inp[1:, :-1]
    )
    contraction_3 += 3 * np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[1:, 1:], inp[1:, 1:], inp[:-1, :-1]
    )
    contraction_3 += 6 * np.einsum(
        "ij,kk,kl,ll->ij", inp[:, :], inp[1:, 1:], inp[1:, :-1], inp[:-1, 1:]
    )
    contraction_3 += 6 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, 1:], inp[:, 1:], inp[1:, 1:], inp[:-1, :-1]
    )
    contraction_3 += 6 * np.einsum(
        "ik,jl,kk,ll->ij", inp[:, 1:], inp[:, :-1], inp[1:, 1:], inp[:-1, 1:]
    )
    contraction_3 += 12 * np.einsum(
        "ik,jk,kl,ll->ij", inp[:, 1:], inp[:, 1:], inp[1:, :-1], inp[:-1, 1:]
    )
    temp = 3 * np.einsum("ik,jl,kk,ll->ij", inp[:, 1:], inp[:, 1:], inp[1:, 1:], inp[:-1, :-1])
    contraction_3 += temp + temp.T
    contraction_3 += 6 * np.einsum(
        "il,jk,kk,ll->ij", inp[:, :-1], inp[:, 1:], inp[1:, 1:], inp[:-1, 1:]
    )

    x7_e_contracted_first = einsum(
        "i,j,k,k,l,l,l->ij", mu, mu, mu, mu, mu[:-1], mu[:-1], mu[1:]
    )
    x7_e_contracted_first += 2 * np.einsum(
        "i,j,k,k,l,ll->ij", mu, mu, mu, mu, mu[:-1], inp[:-1, 1:]
    )
    x7_e_contracted_first += 1 * np.einsum(
        "i,j,k,k,l,ll->ij", mu, mu, mu, mu, mu[1:], inp[:-1, :-1]
    )
    x7_e_contracted_first += 2 * np.einsum(
        "i,j,k,l,l,kl->ij", mu, mu, mu, mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_first += 4 * np.einsum(
        "i,j,k,l,l,kl->ij", mu, mu, mu, mu[:-1], mu[1:], inp[:, :-1]
    )
    temp = 1 * np.einsum("i,k,k,l,l,jl->ij", mu, mu, mu, mu[:-1], mu[:-1], inp[:, 1:])
    x7_e_contracted_first += temp + temp.T
    temp = 2 * np.einsum("i,k,k,l,l,jl->ij", mu, mu, mu, mu[:-1], mu[1:], inp[:, :-1])
    x7_e_contracted_first += temp + temp.T
    x7_e_contracted_first += 1 * np.einsum(
        "k,k,l,l,l,ij->ij", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_first += 1 * np.einsum(
        "i,j,l,l,l,kk->ij", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    temp = 2 * np.einsum("i,k,l,l,l,jk->ij", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :])
    x7_e_contracted_first += temp + temp.T
    x7_e_contracted_first += 2 * np.einsum(
        "i,j,k,klll->ij", mu, mu, mu, fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_first += 2 * np.einsum(
        "i,j,l,kkll->ij", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_first += 1 * np.einsum(
        "i,j,l,kkll->ij", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    temp = 1 * np.einsum("j,k,k,illl->ij", mu, mu, mu, fourth_expecs[:, :-1, :-1, 1:])
    x7_e_contracted_first += temp + temp.T
    temp = 4 * np.einsum("j,k,l,ikll->ij", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:])
    x7_e_contracted_first += temp + temp.T
    temp = 2 * np.einsum("i,k,l,jkll->ij", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1])
    x7_e_contracted_first += temp + temp.T
    temp = 1 * np.einsum("i,l,l,jkkl->ij", mu, mu[:-1], mu[:-1], fourth_expecs[:, :, :, 1:])
    x7_e_contracted_first += temp + temp.T
    temp = 2 * np.einsum("i,l,l,jkkl->ij", mu, mu[:-1], mu[1:], fourth_expecs[:, :, :, :-1])
    x7_e_contracted_first += temp + temp.T
    x7_e_contracted_first += 2 * np.einsum(
        "k,k,l,ijll->ij", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_first += 1 * np.einsum(
        "k,k,l,ijll->ij", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_first += 2 * np.einsum(
        "k,l,l,ijkl->ij", mu, mu[:-1], mu[:-1], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_first += 4 * np.einsum(
        "k,l,l,ijkl->ij", mu, mu[:-1], mu[1:], fourth_expecs[:, :, :, :-1]
    )
    x7_e_contracted_first += 1 * np.einsum(
        "l,l,l,ijkk->ij", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :, :, :]
    )
    temp = 1 * np.einsum("i,jkklll->ij", mu, sixth_expecs[:, :, :, :-1, :-1, 1:])
    x7_e_contracted_first += temp + temp.T
    x7_e_contracted_first += 2 * np.einsum(
        "k,ijklll->ij", mu, sixth_expecs[:, :, :, :-1, :-1, 1:]
    )
    x7_e_contracted_first += 2 * np.einsum(
        "l,ijkkll->ij", mu[:-1], sixth_expecs[:, :, :, :, :-1, 1:]
    )
    x7_e_contracted_first += 1 * np.einsum(
        "l,ijkkll->ij", mu[1:], sixth_expecs[:, :, :, :, :-1, :-1]
    )

    x7_e_contracted_second = einsum(
        "i,j,k,k,l,l,l->ij", mu, mu, mu, mu, mu[1:], mu[1:], mu[1:]
    )
    x7_e_contracted_second += 3 * np.einsum(
        "i,j,k,k,l,ll->ij", mu, mu, mu, mu, mu[1:], inp[1:, 1:]
    )
    x7_e_contracted_second += 6 * np.einsum(
        "i,j,k,l,l,kl->ij", mu, mu, mu, mu[1:], mu[1:], inp[:, 1:]
    )

    temp = 3 * np.einsum("j,k,k,l,l,il->ji", mu, mu, mu, mu[1:], mu[1:], inp[:, 1:])
    x7_e_contracted_second += temp + temp.T

    x7_e_contracted_second += 1 * np.einsum(
        "k,k,l,l,l,ij->ij", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_second += 1 * np.einsum(
        "i,j,l,l,l,kk->ij", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )

    temp = 2 * np.einsum("j,k,l,l,l,ik->ij", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :])
    x7_e_contracted_second += temp + temp.T
    x7_e_contracted_second += 2 * np.einsum(
        "i,j,k,klll->ij", mu, mu, mu, fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_second += 3 * np.einsum(
        "i,j,l,kkll->ij", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    temp = 1 * np.einsum("j,k,k,illl->ij", mu, mu, mu, fourth_expecs[:, 1:, 1:, 1:])
    x7_e_contracted_second += temp + temp.T
    x7_e_contracted_second += 6 * np.einsum(
        "i,k,l,jkll->ij", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    temp = 3 * np.einsum("j,l,l,ikkl->ij", mu, mu[1:], mu[1:], fourth_expecs[:, :, :, 1:])
    x7_e_contracted_second += temp + temp.T
    x7_e_contracted_second += 6 * np.einsum(
        "j,k,l,ikll->ij", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_second += 3 * np.einsum(
        "k,k,l,ijll->ij", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_second += 6 * np.einsum(
        "k,l,l,ijkl->ij", mu, mu[1:], mu[1:], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_second += 1 * np.einsum(
        "l,l,l,ijkk->ij", mu[1:], mu[1:], mu[1:], fourth_expecs[:, :, :, :]
    )
    temp = 1 * np.einsum("j,ikklll->ij", mu, sixth_expecs[:, :, :, 1:, 1:, 1:])
    x7_e_contracted_second += temp + temp.T
    x7_e_contracted_second += 2 * np.einsum(
        "k,ijklll->ij", mu, sixth_expecs[:, :, :, 1:, 1:, 1:]
    )
    x7_e_contracted_second += 3 * np.einsum(
        "l,ijkkll->ij", mu[1:], sixth_expecs[:, :, :, :, 1:, 1:]
    )

    x7_e_contracted_firstV = einsum(
        "i,k,k,k,l,l,l->i", mu, mu[1:], mu[1:], mu[1:], mu[1:], mu[1:], mu[1:]
    )
    x7_e_contracted_firstV += 9 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[1:], mu[1:], mu[1:], mu[1:], inp[1:, 1:]
    )
    x7_e_contracted_firstV += 6 * np.einsum(
        "k,k,l,l,l,ik->i", mu[1:], mu[1:], mu[1:], mu[1:], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_firstV += 6 * np.einsum(
        "i,k,l,l,l,kk->i", mu, mu[1:], mu[1:], mu[1:], mu[1:], inp[1:, 1:]
    )
    x7_e_contracted_firstV += 3 * np.einsum(
        "i,k,k,klll->i", mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 9 * np.einsum(
        "i,k,l,kkll->i", mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 3 * np.einsum(
        "i,l,l,kkkl->i", mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 2 * np.einsum(
        "k,k,k,illl->i", mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 9 * np.einsum(
        "k,k,l,ikll->i", mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 9 * np.einsum(
        "k,l,l,ikkl->i", mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 1 * np.einsum(
        "i,kkklll->i", mu, sixth_expecs[1:, 1:, 1:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 3 * np.einsum(
        "k,ikklll->i", mu[1:], sixth_expecs[:, 1:, 1:, 1:, 1:, 1:]
    )
    x7_e_contracted_firstV += 3 * np.einsum(
        "l,ikkkll->i", mu[1:], sixth_expecs[:, 1:, 1:, 1:, 1:, 1:]
    )

    x7_e_contracted_secondV = einsum(
        "i,k,k,k,l,l,l->i", mu, mu[:-1], mu[:-1], mu[1:], mu[:-1], mu[:-1], mu[1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,k,k,l,ll->i", mu, mu[:-1], mu[:-1], mu[1:], mu[:-1], inp[:-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,k,k,k,l,ll->i", mu, mu[:-1], mu[:-1], mu[1:], mu[1:], inp[:-1, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[:-1], mu[:-1], mu[:-1], mu[:-1], inp[1:, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[:-1], mu[:-1], mu[:-1], mu[1:], inp[1:, :-1]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[:-1], mu[1:], mu[:-1], mu[:-1], inp[:-1, 1:]
    )
    x7_e_contracted_secondV += 4 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[:-1], mu[1:], mu[:-1], mu[1:], inp[:-1, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "k,k,k,l,l,il->i", mu[:-1], mu[:-1], mu[1:], mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,k,k,l,l,il->i", mu[:-1], mu[:-1], mu[1:], mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,k,l,l,l,ik->i", mu[:-1], mu[1:], mu[:-1], mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,l,l,l,kk->i", mu, mu[:-1], mu[:-1], mu[:-1], mu[1:], inp[:-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,k,l,l,l,kk->i", mu, mu[1:], mu[:-1], mu[:-1], mu[1:], inp[:-1, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "k,k,l,l,l,ik->i", mu[:-1], mu[:-1], mu[:-1], mu[:-1], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,k,k,klll->i", mu, mu[:-1], mu[:-1], fourth_expecs[1:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,k,klll->i", mu, mu[:-1], mu[1:], fourth_expecs[:-1, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 4 * np.einsum(
        "i,k,l,kkll->i", mu, mu[:-1], mu[:-1], fourth_expecs[:-1, 1:, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,l,kkll->i", mu, mu[:-1], mu[1:], fourth_expecs[:-1, 1:, :-1, :-1]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,k,l,kkll->i", mu, mu[1:], mu[:-1], fourth_expecs[:-1, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,k,l,kkll->i", mu, mu[1:], mu[1:], fourth_expecs[:-1, :-1, :-1, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,l,l,kkkl->i", mu, mu[:-1], mu[:-1], fourth_expecs[:-1, :-1, 1:, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "i,l,l,kkkl->i", mu, mu[:-1], mu[1:], fourth_expecs[:-1, :-1, 1:, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "k,k,k,illl->i", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,k,l,ikll->i", mu[:-1], mu[:-1], mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "k,k,l,ikll->i", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    x7_e_contracted_secondV += 4 * np.einsum(
        "k,k,l,ikll->i", mu[:-1], mu[1:], mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,k,l,ikll->i", mu[:-1], mu[1:], mu[1:], fourth_expecs[:, :-1, :-1, :-1]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,l,l,ikkl->i", mu[:-1], mu[:-1], mu[:-1], fourth_expecs[:, :-1, 1:, 1:]
    )
    x7_e_contracted_secondV += 4 * np.einsum(
        "k,l,l,ikkl->i", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :-1, 1:, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "k,l,l,ikkl->i", mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,l,l,ikkl->i", mu[1:], mu[:-1], mu[1:], fourth_expecs[:, :-1, :-1, :-1]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "l,l,l,ikkk->i", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "i,kkklll->i", mu, sixth_expecs[:-1, :-1, 1:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "k,ikklll->i", mu[:-1], sixth_expecs[:, :-1, 1:, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "k,ikklll->i", mu[1:], sixth_expecs[:, :-1, :-1, :-1, :-1, 1:]
    )
    x7_e_contracted_secondV += 2 * np.einsum(
        "l,ikkkll->i", mu[:-1], sixth_expecs[:, :-1, :-1, 1:, :-1, 1:]
    )
    x7_e_contracted_secondV += 1 * np.einsum(
        "l,ikkkll->i", mu[1:], sixth_expecs[:, :-1, :-1, 1:, :-1, :-1]
    )

    x7_e_contracted_thirdV = einsum(
        "i,k,k,k,l,l,l->i", mu, mu[1:], mu[1:], mu[1:], mu[:-1], mu[:-1], mu[1:]
    )
    x7_e_contracted_thirdV += 2 * np.einsum(
        "i,k,k,k,l,ll->i", mu, mu[1:], mu[1:], mu[1:], mu[:-1], inp[:-1, 1:]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "i,k,k,k,l,ll->i", mu, mu[1:], mu[1:], mu[1:], mu[1:], inp[:-1, :-1]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[1:], mu[1:], mu[:-1], mu[:-1], inp[1:, 1:]
    )
    x7_e_contracted_thirdV += 6 * np.einsum(
        "i,k,k,l,l,kl->i", mu, mu[1:], mu[1:], mu[:-1], mu[1:], inp[1:, :-1]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "k,k,k,l,l,il->i", mu[1:], mu[1:], mu[1:], mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_thirdV += 2 * np.einsum(
        "k,k,k,l,l,il->i", mu[1:], mu[1:], mu[1:], mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "k,k,l,l,l,ik->i", mu[1:], mu[1:], mu[:-1], mu[:-1], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "i,k,l,l,l,kk->i", mu, mu[1:], mu[:-1], mu[:-1], mu[1:], inp[1:, 1:]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "i,k,k,klll->i", mu, mu[1:], mu[1:], fourth_expecs[1:, :-1, :-1, 1:]
    )
    x7_e_contracted_thirdV += 6 * np.einsum(
        "i,k,l,kkll->i", mu, mu[1:], mu[:-1], fourth_expecs[1:, 1:, :-1, 1:]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "i,k,l,kkll->i", mu, mu[1:], mu[1:], fourth_expecs[1:, 1:, :-1, :-1]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "i,l,l,kkkl->i", mu, mu[:-1], mu[:-1], fourth_expecs[1:, 1:, 1:, 1:]
    )
    x7_e_contracted_thirdV += 2 * np.einsum(
        "i,l,l,kkkl->i", mu, mu[:-1], mu[1:], fourth_expecs[1:, 1:, 1:, :-1]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "k,k,k,illl->i", mu[1:], mu[1:], mu[1:], fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_thirdV += 6 * np.einsum(
        "k,k,l,ikll->i", mu[1:], mu[1:], mu[:-1], fourth_expecs[:, 1:, :-1, 1:]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "k,k,l,ikll->i", mu[1:], mu[1:], mu[1:], fourth_expecs[:, 1:, :-1, :-1]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "k,l,l,ikkl->i", mu[1:], mu[:-1], mu[:-1], fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_thirdV += 6 * np.einsum(
        "k,l,l,ikkl->i", mu[1:], mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, :-1]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "l,l,l,ikkk->i", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "i,kkklll->i", mu, sixth_expecs[1:, 1:, 1:, :-1, :-1, 1:]
    )
    x7_e_contracted_thirdV += 3 * np.einsum(
        "k,ikklll->i", mu[1:], sixth_expecs[:, 1:, 1:, :-1, :-1, 1:]
    )
    x7_e_contracted_thirdV += 2 * np.einsum(
        "l,ikkkll->i", mu[:-1], sixth_expecs[:, 1:, 1:, 1:, :-1, 1:]
    )
    x7_e_contracted_thirdV += 1 * np.einsum(
        "l,ikkkll->i", mu[1:], sixth_expecs[:, 1:, 1:, 1:, :-1, :-1]
    )

    x7_e_contracted_fourth = einsum(
        "i,j,k,l,m,m,m->ijkl", mu, mu, mu, mu, mu[:-1], mu[:-1], mu[1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,j,k,l,m,mm->ijkl", mu, mu, mu, mu, mu[:-1], inp[:-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,k,l,m,mm->ijkl", mu, mu, mu, mu, mu[1:], inp[:-1, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,k,m,m,lm->ijkl", mu, mu, mu, mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,j,k,m,m,lm->ijkl", mu, mu, mu, mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,l,m,m,km->ijkl", mu, mu, mu, mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,j,l,m,m,km->ijkl", mu, mu, mu, mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,k,l,m,m,jm->ijkl", mu, mu, mu, mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,k,l,m,m,jm->ijkl", mu, mu, mu, mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,k,l,m,m,im->ijkl", mu, mu, mu, mu[:-1], mu[:-1], inp[:, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "j,k,l,m,m,im->ijkl", mu, mu, mu, mu[:-1], mu[1:], inp[:, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "k,l,m,m,m,ij->ijkl", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,m,m,m,kl->ijkl", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,k,m,m,m,jl->ijkl", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,l,m,m,m,jk->ijkl", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,k,m,m,m,il->ijkl", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,l,m,m,m,ik->ijkl", mu, mu, mu[:-1], mu[:-1], mu[1:], inp[:, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,k,lmmm->ijkl", mu, mu, mu, fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,l,kmmm->ijkl", mu, mu, mu, fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,j,m,klmm->ijkl", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,j,m,klmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,k,l,jmmm->ijkl", mu, mu, mu, fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,k,m,jlmm->ijkl", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,k,m,jlmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,l,m,jkmm->ijkl", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,l,m,jkmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,m,m,jklm->ijkl", mu, mu[:-1], mu[:-1], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "i,m,m,jklm->ijkl", mu, mu[:-1], mu[1:], fourth_expecs[:, :, :, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,k,l,immm->ijkl", mu, mu, mu, fourth_expecs[:, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "j,k,m,ilmm->ijkl", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,k,m,ilmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "j,l,m,ikmm->ijkl", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,l,m,ikmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,m,m,iklm->ijkl", mu, mu[:-1], mu[:-1], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "j,m,m,iklm->ijkl", mu, mu[:-1], mu[1:], fourth_expecs[:, :, :, :-1]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "k,l,m,ijmm->ijkl", mu, mu, mu[:-1], fourth_expecs[:, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "k,l,m,ijmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, :-1, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "k,m,m,ijlm->ijkl", mu, mu[:-1], mu[:-1], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "k,m,m,ijlm->ijkl", mu, mu[:-1], mu[1:], fourth_expecs[:, :, :, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "l,m,m,ijkm->ijkl", mu, mu[:-1], mu[:-1], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "l,m,m,ijkm->ijkl", mu, mu[:-1], mu[1:], fourth_expecs[:, :, :, :-1]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "m,m,m,ijkl->ijkl", mu[:-1], mu[:-1], mu[1:], fourth_expecs[:, :, :, :]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "i,jklmmm->ijkl", mu, sixth_expecs[:, :, :, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "j,iklmmm->ijkl", mu, sixth_expecs[:, :, :, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "k,ijlmmm->ijkl", mu, sixth_expecs[:, :, :, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "l,ijkmmm->ijkl", mu, sixth_expecs[:, :, :, :-1, :-1, 1:]
    )
    x7_e_contracted_fourth += 2 * np.einsum(
        "m,ijklmm->ijkl", mu[:-1], sixth_expecs[:, :, :, :, :-1, 1:]
    )
    x7_e_contracted_fourth += 1 * np.einsum(
        "m,ijklmm->ijkl", mu[1:], sixth_expecs[:, :, :, :, :-1, :-1]
    )
    x7_e_contracted_fifth = einsum(
        "i,j,k,l,m,m,m->ijkl", mu, mu, mu, mu, mu[1:], mu[1:], mu[1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,j,k,l,m,mm->ijkl", mu, mu, mu, mu, mu[1:], inp[1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,j,k,m,m,lm->ijkl", mu, mu, mu, mu[1:], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,j,l,m,m,km->ijkl", mu, mu, mu, mu[1:], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,k,l,m,m,jm->ijkl", mu, mu, mu, mu[1:], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "j,k,l,m,m,im->ijkl", mu, mu, mu, mu[1:], mu[1:], inp[:, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "k,l,m,m,m,ij->ijkl", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,j,m,m,m,kl->ijkl", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,k,m,m,m,jl->ijkl", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,l,m,m,m,jk->ijkl", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "j,k,m,m,m,il->ijkl", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "j,l,m,m,m,ik->ijkl", mu, mu, mu[1:], mu[1:], mu[1:], inp[:, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,j,k,lmmm->ijkl", mu, mu, mu, fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,j,l,kmmm->ijkl", mu, mu, mu, fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,j,m,klmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,k,l,jmmm->ijkl", mu, mu, mu, fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,k,m,jlmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,l,m,jkmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "i,m,m,jklm->ijkl", mu, mu[1:], mu[1:], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "j,k,l,immm->ijkl", mu, mu, mu, fourth_expecs[:, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "j,k,m,ilmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "j,l,m,ikmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "j,m,m,iklm->ijkl", mu, mu[1:], mu[1:], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "k,l,m,ijmm->ijkl", mu, mu, mu[1:], fourth_expecs[:, :, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "k,m,m,ijlm->ijkl", mu, mu[1:], mu[1:], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "l,m,m,ijkm->ijkl", mu, mu[1:], mu[1:], fourth_expecs[:, :, :, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "m,m,m,ijkl->ijkl", mu[1:], mu[1:], mu[1:], fourth_expecs[:, :, :, :]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "i,jklmmm->ijkl", mu, sixth_expecs[:, :, :, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "j,iklmmm->ijkl", mu, sixth_expecs[:, :, :, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "k,ijlmmm->ijkl", mu, sixth_expecs[:, :, :, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 1 * np.einsum(
        "l,ijkmmm->ijkl", mu, sixth_expecs[:, :, :, 1:, 1:, 1:]
    )
    x7_e_contracted_fifth += 3 * np.einsum(
        "m,ijklmm->ijkl", mu[1:], sixth_expecs[:, :, :, :, 1:, 1:]
    )

    return (
        contraction_1,
        contraction_2,
        contraction_3,
        x7_e_contracted_first,
        x7_e_contracted_second,
        x7_e_contracted_firstV,
        x7_e_contracted_secondV,
        x7_e_contracted_thirdV,
        x7_e_contracted_fourth,
        x7_e_contracted_fifth,
    )


if __name__ == "__main__":
    dim = 3
    matrix = np.array([[0.1, 0.2, 0.3, 0.7]]).reshape(2, 2)
    matrix = np.random.rand(dim, dim)
    cmatrix = np.array([[0.1, 0.2, 0.3, 0.7]]).reshape(2, 2)
    cmatrix = np.random.rand(dim, dim)
    matrix = 1 / 100 * matrix.T @ matrix + 1j * (cmatrix + cmatrix.T)
    matrix = np.eye(2) * np.sqrt(2)
    input_matrix = inp = matrix / 2
    mu = np.array([0.1 + 0.1j, -0.2 - 0.2j])
    mu = np.random.rand(dim) + 1j * np.random.rand(dim)
    mu = np.zeros(2)
    start = time.time()
    sintegrals = calculate_full_expectation_value(
        matrix,
        mu,
        order=8,
    )
    x8_contraction1 = einsum("kliiijjj->kl", sintegrals[:, :, 1:, 1:, 1:, 1:, 1:, 1:])
    x8_contraction2 = einsum("kliiijjj->kl", sintegrals[:, :, :-1, :-1, 1:, :-1, :-1, 1:])
    x8_contraction3 = einsum("kliiijjj->kl", sintegrals[:, :, 1:, 1:, 1:, 0:-1, 0:-1, 1:])
    sintegrals2 = calculate_full_expectation_value(
        matrix,
        mu,
        order=7,
    )
    x7_e_contracted_1 = einsum("kliijjj->kl", sintegrals2[:, :, :, :, :-1, :-1, 1:])
    x7_e_contracted_2 = einsum("kliijjj->kl", sintegrals2[:, :, :, :, 1:, 1:, 1:])
    x7_e_contracted_3 = einsum("ikkklll->i", sintegrals2[:, 1:, 1:, 1:, 1:, 1:, 1:])
    x7_e_contracted_4 = einsum("kiiijjj->k", sintegrals2[:, :-1, :-1, 1:, :-1, :-1, 1:])
    x7_e_contracted_5 = einsum("kiiijjj->k", sintegrals2[:, 1:, 1:, 1:, 0:-1, 0:-1, 1:])
    x7_e_contracted_6 = einsum("ijklmmm->ijkl", sintegrals2[:, :, :, :, 0:-1, 0:-1, 1:])
    x7_e_contracted_7 = einsum("ijklmmm->ijkl", sintegrals2[:, :, :, :, 1:, 1:, 1:])

    end = time.time()

    start = time.time()
    (
        x8_contraction1_alt,
        x8_contraction2_alt,
        x8_contraction3_alt,
        x7_e_contracted_alt_1,
        x7_e_contracted_alt_2,
        x7_e_contracted_alt_3,
        x7_e_contracted_alt_4,
        x7_e_contracted_alt_5,
        x7_e_contracted_alt_6,
        x7_e_contracted_alt_7,
    ) = get_x7_x8_contraction(matrix, mu)
    end = time.time()
    tots = []
    tots.append(sum(abs(x7_e_contracted_1 - x7_e_contracted_alt_1).flatten()))
    tots.append(sum(abs(x7_e_contracted_2 - x7_e_contracted_alt_2).flatten()))
    tots.append(sum(abs(x7_e_contracted_3 - x7_e_contracted_alt_3).flatten()))
    tots.append(sum(abs(x7_e_contracted_4 - x7_e_contracted_alt_4).flatten()))
    tots.append(sum(abs(x7_e_contracted_5 - x7_e_contracted_alt_5).flatten()))
    tots.append(sum(abs(x7_e_contracted_6 - x7_e_contracted_alt_6).flatten()))

    tots.append(sum(abs(x7_e_contracted_7 - x7_e_contracted_alt_7).flatten()))
    tots.append(sum(abs(x8_contraction1_alt - x8_contraction1).flatten()))
    tots.append(sum(abs(x8_contraction2_alt - x8_contraction2).flatten()))
    tots.append(sum(abs(x8_contraction3_alt - x8_contraction3).flatten()))
    print("Deviation from by-hand-calculation: %e" % sum(tots))
