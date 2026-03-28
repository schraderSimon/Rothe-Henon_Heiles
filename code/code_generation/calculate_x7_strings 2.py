def countX(lst, x):
    count = 0
    for ele in lst:
        if ele == x:
            count = count + 1
    return count


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
fullstrings = []
for string in strings7_5:
    pre_arrow = string.split("->")[0]
    pre_arrow = pre_arrow.split(",")[0] + pre_arrow.split(",")[1]
    new = pre_arrow  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i < 5:
            newstring += ","
    newstring += "->ijkl"
strings7_5_upd = [
    "i,j,k,l,m,no->ijkl",
    "i,j,k,l,n,mo->ijkl",
    "i,j,k,l,o,mn->ijkl",
    "i,j,k,m,n,lo->ijkl",
    "i,j,k,m,o,ln->ijkl",
    "i,j,k,n,o,lm->ijkl",
    "i,j,l,m,n,ko->ijkl",
    "i,j,l,m,o,kn->ijkl",
    "i,j,l,n,o,km->ijkl",
    "i,k,l,m,n,jo->ijkl",
    "i,k,l,m,o,jn->ijkl",
    "i,k,l,n,o,jm->ijkl",
    "j,k,l,m,n,io->ijkl",
    "j,k,l,m,o,in->ijkl",
    "j,k,l,n,o,im->ijkl",
    "k,l,m,n,o,ij->ijkl",
    "i,j,m,n,o,kl->ijkl",
    "i,k,m,n,o,jl->ijkl",
    "i,l,m,n,o,jk->ijkl",
    "j,k,m,n,o,il->ijkl",
    "j,l,m,n,o,ik->ijkl",
]
all_elems = []
for i, string in enumerate(strings7_5_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(5):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            adders.append("[1:]")
        elif fullstrings[i][x] == "o":
            adders.append("[1:]")
        else:
            adders.append("")
    extraadder = "["
    for x in range(5, 7):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            extraadder += "1:,"
        elif fullstrings[i][x] == "o":
            extraadder += "1:,"
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
        + ",inp"
        + extraadder
        + ")"
    )
    replaced = finalstring.replace("n", "m").replace("o", "m").replace("imp", "inp")
    finalstring = "x7_e_contracted_fifth+=np.einsum(" + replaced
    all_elems.append(finalstring)
    # print(finalstring)
mylist = list(dict.fromkeys(all_elems))
for elem in mylist:
    count = countX(all_elems, elem)
    elem_splitted = elem.split("+=")
    print("%s+=%d*%s" % (elem_splitted[0], count, elem_splitted[1]))
fullstrings = []
for string in strings7_3:
    pre_arrow = string.split("->")[0]
    pre_arrow = pre_arrow.split(",")[0] + pre_arrow.split(",")[1]
    new = pre_arrow  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i < 3:
            newstring += ","
    newstring += "->ijkl"

strings7_3_upd = [
    "i,j,k,lmno->ijkl",
    "i,j,l,kmno->ijkl",
    "i,j,m,klno->ijkl",
    "i,j,n,klmo->ijkl",
    "i,j,o,klmn->ijkl",
    "i,k,l,jmno->ijkl",
    "i,k,m,jlno->ijkl",
    "i,k,n,jlmo->ijkl",
    "i,k,o,jlmn->ijkl",
    "i,l,m,jkno->ijkl",
    "i,l,n,jkmo->ijkl",
    "i,l,o,jkmn->ijkl",
    "i,m,n,jklo->ijkl",
    "i,m,o,jkln->ijkl",
    "i,n,o,jklm->ijkl",
    "j,k,l,imno->ijkl",
    "j,k,m,ilno->ijkl",
    "j,k,n,ilmo->ijkl",
    "j,k,o,ilmn->ijkl",
    "j,l,m,ikno->ijkl",
    "j,l,n,ikmo->ijkl",
    "j,l,o,ikmn->ijkl",
    "j,m,n,iklo->ijkl",
    "j,m,o,ikln->ijkl",
    "j,n,o,iklm->ijkl",
    "k,l,m,ijno->ijkl",
    "k,l,n,ijmo->ijkl",
    "k,l,o,ijmn->ijkl",
    "k,m,n,ijlo->ijkl",
    "k,m,o,ijln->ijkl",
    "k,n,o,ijlm->ijkl",
    "l,m,n,ijko->ijkl",
    "l,m,o,ijkn->ijkl",
    "l,n,o,ijkm->ijkl",
    "m,n,o,ijkl->ijkl",
]
all_elems = []
for i, string in enumerate(strings7_3_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(3):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            adders.append("[1:]")
        elif fullstrings[i][x] == "o":
            adders.append("[1:]")
        else:
            adders.append("")
    extraadder = "["
    for x in range(3, 7):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            extraadder += "1:,"
        elif fullstrings[i][x] == "o":
            extraadder += "1:,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    returnstring = "'%s',mu" % string + adders[0]
    returnstring = returnstring.replace("n", "m").replace("o", "m").replace("imp", "inp")
    finalstring = returnstring + ",mu" + adders[1] + ",mu" + adders[2] + ",fourth_expecs" + extraadder + ")"
    finalstring = "x7_e_contracted_fifth+=np.einsum(" + finalstring
    all_elems.append(finalstring)
    # print(finalstring)
mylist = list(dict.fromkeys(all_elems))
for elem in mylist:
    count = countX(all_elems, elem)
    elem_splitted = elem.split("+=")
    print("%s+=%d*%s" % (elem_splitted[0], count, elem_splitted[1]))

fullstrings = []
for string in strings7_1:
    pre_arrow = string.split("->")[0]
    pre_arrow = pre_arrow.split(",")[0] + pre_arrow.split(",")[1]
    new = pre_arrow  # .replace("l","k").replace("m","k").replace("n","l").replace("o","l").replace("p","l")
    fullstrings.append(new)
    newstring = ""
    for i, elem in enumerate(new):
        newstring += elem
        if i < 1:
            newstring += ","
    newstring += "->ijkl"

strings7_1_upd = [
    "i,jklmno->ijkl",
    "j,iklmno->ijkl",
    "k,ijlmno->ijkl",
    "l,ijkmno->ijkl",
    "m,ijklno->ijkl",
    "n,ijklmo->ijkl",
    "o,ijklmn->ijkl",
]

all_elems = []
for i, string in enumerate(strings7_1_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(1):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            adders.append("[1:]")
        elif fullstrings[i][x] == "o":
            adders.append("[1:]")
        else:
            adders.append("")
    extraadder = "["
    for x in range(1, 7):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            extraadder += "1:,"
        elif fullstrings[i][x] == "o":
            extraadder += "1:,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    returnstring = "'%s',mu" % string + adders[0]
    returnstring = returnstring.replace("n", "m").replace("o", "m").replace("imp", "inp")
    finalstring = returnstring + ",sixth_expecs" + extraadder + ")"
    finalstring = "x7_e_contracted_fifth+=np.einsum(" + finalstring
    all_elems.append(finalstring)
    # print(finalstring)
mylist = list(dict.fromkeys(all_elems))
for elem in mylist:
    count = countX(all_elems, elem)
    elem_splitted = elem.split("+=")
    print("%s+=%d*%s" % (elem_splitted[0], count, elem_splitted[1]))
