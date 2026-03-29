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
fullstrings = []
for string in strings8_6:
    pre_arrow = string.split("->")[0]
    pre_arrow = pre_arrow.split(",")[0] + pre_arrow.split(",")[1]
    new = pre_arrow  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i < 6:
            newstring += ","
    newstring += "->ij"
    print("'%s'," % newstring)
strings8_6_upd = [
    "i,j,k,l,m,n,op->ij",
    "i,j,k,l,m,o,np->ij",
    "i,j,k,l,m,p,no->ij",
    "i,j,k,l,n,o,mp->ij",
    "i,j,k,l,n,p,mo->ij",
    "i,j,k,l,o,p,mn->ij",
    "i,j,k,m,n,o,lp->ij",
    "i,j,k,m,n,p,lo->ij",
    "i,j,k,m,o,p,ln->ij",
    "i,j,k,n,o,p,lm->ij",
    "i,j,l,n,o,m,kp->ij",
    "i,j,l,m,n,p,ko->ij",
    "i,j,l,m,o,p,kn->ij",
    "i,j,l,n,o,p,km->ij",
    "i,j,m,n,o,p,kl->ij",
    "i,k,l,m,o,p,jn->ij",
    "i,k,l,n,o,p,jm->ij",
    "i,k,m,n,o,p,jl->ij",
    "i,l,m,n,o,p,jk->ij",
    "j,k,l,m,o,p,in->ij",
    "j,k,l,n,o,p,im->ij",
    "j,k,m,n,o,p,il->ij",
    "j,l,m,n,o,p,ik->ij",
    "k,l,m,n,o,p,ij->ij",
    "j,k,l,m,n,o,ip->ij",
    "j,k,l,m,n,p,io->ij",
    "i,k,l,m,n,o,jp->ij",
    "i,k,l,m,n,p,jo->ij",
]
for i, string in enumerate(strings8_6_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(6):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            adders.append("[1:]")
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            adders.append("[:-1]")

        else:
            adders.append("")
    extraadder = "["
    for x in range(6, 8):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            extraadder += "1:,"
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            extraadder += ":-1,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    returnstring = "'%s',mu" % string + adders[0]
    finalstring = (
        returnstring
        + ",mu"
        + adders[1]
        + ",mu"
        + adders[2]
        + ",mu"
        + adders[3]
        + ",mu"
        + adders[4]
        + ",mu"
        + adders[5]
        + ",inp"
        + extraadder
        + ")"
    )
    finalstring = "contraction_3+=np.einsum(" + finalstring.replace("inp", "y").replace(
        "mu", "x"
    ).replace("l", "k").replace("m", "k").replace("n", "l").replace("o", "l").replace(
        "p", "l"
    ).replace(
        "x", "mu"
    ).replace(
        "y", "inp"
    )
    print(finalstring)
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

fullstrings = []
for string in strings8_4:
    pre_arrow = string.split("->")[0]
    pre_arrow = pre_arrow.split(",")[0] + pre_arrow.split(",")[1]
    new = pre_arrow  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i < 4:
            newstring += ","
    newstring += "->ij"
    print("'%s'," % newstring)
strings8_4_upd = [
    "i,j,k,l,mnop->ij",
    "i,j,k,m,lnop->ij",
    "i,j,k,n,lmop->ij",
    "i,j,k,o,lmnp->ij",
    "i,j,k,p,lmno->ij",
    "i,j,l,m,knop->ij",
    "i,j,l,n,kmop->ij",
    "i,j,l,o,kmnp->ij",
    "i,j,l,p,kmno->ij",
    "i,j,m,n,klop->ij",
    "i,j,m,o,klnp->ij",
    "i,j,m,p,klno->ij",
    "i,j,n,o,klmp->ij",
    "i,j,n,p,klmo->ij",
    "i,j,o,p,klmn->ij",
    "i,k,l,m,jnop->ij",
    "i,k,l,n,jmop->ij",
    "i,k,l,o,jmnp->ij",
    "i,k,l,p,jmno->ij",
    "i,k,m,n,jlop->ij",
    "i,k,m,o,jlnp->ij",
    "i,k,m,p,jlno->ij",
    "i,k,n,o,jlmp->ij",
    "i,k,n,p,jlmo->ij",
    "i,k,o,p,jlmn->ij",
    "j,l,o,p,ikmn->ij",
    "i,l,o,p,jkmn->ij",
    "m,n,o,p,ijkl->ij",
    "i,m,o,p,jkln->ij",
    "i,n,o,p,jklm->ij",
    "k,l,o,p,ijmn->ij",
    "j,k,o,p,ilmn->ij",
    "l,m,o,p,ijkn->ij",
    "l,n,o,p,ijkm->ij",
    "k,n,o,p,ijlm->ij",
    "j,m,o,p,ikln->ij",
    "j,n,o,p,iklm->ij",
    "l,m,k,n,ijop->ij",
    "l,m,k,o,ijnp->ij",
    "l,m,k,p,ijno->ij",
    "l,m,n,o,ijkp->ij",
    "l,m,n,p,ijko->ij",
    "l,m,j,k,inop->ij",
    "l,m,i,n,jkop->ij",
    "l,m,i,o,jknp->ij",
    "l,m,i,p,jkno->ij",
    "l,m,j,n,ikop->ij",
    "l,m,j,o,iknp->ij",
    "l,m,j,p,ikno->ij",
    "j,l,n,o,ikmp->ij",
    "j,l,n,p,ikmo->ij",
    "i,m,n,p,jklo->ij",
    "m,j,n,o,iklp->ij",
    "k,l,n,o,ijmp->ij",
    "k,l,n,p,ijmo->ij",
    "j,k,l,n,imop->ij",
    "j,k,l,o,imnp->ij",
    "j,k,l,p,imno->ij",
    "j,k,m,n,ilop->ij",
    "j,k,m,o,ilnp->ij",
    "j,k,m,p,ilno->ij",
    "j,k,n,o,ilmp->ij",
    "j,k,n,p,ilmo->ij",
    "i,l,n,o,jkmp->ij",
    "i,l,n,p,jkmo->ij",
    "i,m,n,o,jklp->ij",
    "k,m,n,o,ijlp->ij",
    "k,m,n,p,ijlo->ij",
    "k,m,o,p,ijln->ij",
    "m,j,n,p,iklo->ij",
]


for i, string in enumerate(strings8_4_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(4):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            adders.append("[1:]")
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            adders.append("[:-1]")

        else:
            adders.append("")
    extraadder = "["
    for x in range(4, 8):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            extraadder += "1:,"
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            extraadder += ":-1,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    string = (
        string.replace("inp", "y")
        .replace("mu", "x")
        .replace("l", "k")
        .replace("m", "k")
        .replace("n", "l")
        .replace("o", "l")
        .replace("p", "l")
        .replace("x", "mu")
        .replace("y", "inp")
    )
    returnstring = "'%s',mu" % string + adders[0]
    finalstring = (
        returnstring
        + ",mu"
        + adders[1]
        + ",mu"
        + adders[2]
        + ",mu"
        + adders[3]
        + ",fourh_expecs"
        + extraadder
        + ")"
    )
    finalstring = "contraction_3+=np.einsum(" + finalstring
    print(finalstring)
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
fullstrings = []
for string in strings8_2:
    pre_arrow = string.split("->")[0]
    pre_arrow = pre_arrow.split(",")[0] + pre_arrow.split(",")[1]
    new = pre_arrow  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i < 2:
            newstring += ","
    newstring += "->ij"
    print("'%s'," % newstring)
strings8_2_upd = [
    "i,j,klmnop->ij",
    "i,k,jlmnop->ij",
    "i,l,jkmnop->ij",
    "i,m,jklnop->ij",
    "i,n,jklmop->ij",
    "i,o,jklmnp->ij",
    "i,p,jklmno->ij",
    "j,k,ilmnop->ij",
    "j,l,ikmnop->ij",
    "j,m,iklnop->ij",
    "j,n,iklmop->ij",
    "j,o,iklmnp->ij",
    "j,p,iklmno->ij",
    "k,l,ijmnop->ij",
    "k,m,ijlnop->ij",
    "k,n,ijlmop->ij",
    "k,o,ijlmnp->ij",
    "k,p,ijlmno->ij",
    "l,m,ijknop->ij",
    "l,n,ijkmop->ij",
    "l,o,ijkmnp->ij",
    "l,p,ijkmno->ij",
    "m,n,ijklop->ij",
    "m,o,ijklnp->ij",
    "m,p,ijklno->ij",
    "n,o,ijklmp->ij",
    "n,p,ijklmo->ij",
    "o,p,ijklmn->ij",
]
for i, string in enumerate(strings8_2_upd[:]):
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(2):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            adders.append("[1:]")
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            adders.append("[:-1]")

        else:
            adders.append("")
    extraadder = "["
    for x in range(2, 8):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            extraadder += "1:,"
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            extraadder += ":-1,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    string = (
        string.replace("inp", "y")
        .replace("mu", "x")
        .replace("l", "k")
        .replace("m", "k")
        .replace("n", "l")
        .replace("o", "l")
        .replace("p", "l")
        .replace("x", "mu")
        .replace("y", "inp")
    )
    returnstring = "'%s',mu" % string + adders[0]
    finalstring = returnstring + ",mu" + adders[1] + ",sixth_expecs" + extraadder + ")"
    finalstring = "contraction_3+=np.einsum(" + finalstring
    print(finalstring)
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
fullstrings = []
assstrings = []
for string in eight_partition_strings:
    new = string  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i % 2 == 1 and i > 0:
            newstring += ","
    newstring = newstring[:-1]
    newstring += "->ij"
    print("'%s'," % newstring)
    assstrings.append(newstring)
print(assstrings)
all_elems = []
for i, string in enumerate(fullstrings):
    extraadder = ""
    for x in range(0, 8):
        if (
            fullstrings[i][x] == "k"
            or fullstrings[i][x] == "l"
            or fullstrings[i][x] == "m"
            or fullstrings[i][x] == "p"
        ):
            extraadder += "1:,"
        elif fullstrings[i][x] == "n" or fullstrings[i][x] == "o":
            extraadder += ":-1,"
        else:
            extraadder += ":,"

    extraadder = extraadder[:-1]
    terms = extraadder.split(",")
    v1 = "[" + terms[0] + "," + terms[1] + "]"
    v2 = "[" + terms[2] + "," + terms[3] + "]"
    v3 = "[" + terms[4] + "," + terms[5] + "]"
    v4 = "[" + terms[6] + "," + terms[7] + "]"
    returnstring = (
        "contraction_3+=np.einsum('"
        + assstrings[i]
        .replace("l", "k")
        .replace("m", "k")
        .replace("n", "l")
        .replace("o", "l")
        .replace("p", "l")
        + "',inp"
        + v1
        + ",inp"
        + v2
        + ",inp"
        + v3
        + ",inp"
        + v4
        + ")"
    )
    all_elems.append(returnstring)
    print(returnstring)
mylist = list(dict.fromkeys(all_elems))


def countX(lst, x):
    count = 0
    for ele in lst:
        if ele == x:
            count = count + 1
    return count


for elem in mylist:
    count = countX(all_elems, elem)
    elem_splitted = elem.split("+=")
    print("%s+=%d*%s" % (elem_splitted[0], count, elem_splitted[1]))
