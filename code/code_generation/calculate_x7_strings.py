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
    newstring += "->i"
    # print("'%s',"%newstring)
strings7_5_upd = [
    "i,j,k,l,m,no->i",
    "i,j,k,l,n,mo->i",
    "i,j,k,l,o,mn->i",
    "i,j,k,m,n,lo->i",
    "i,j,k,m,o,ln->i",
    "i,j,k,n,o,lm->i",
    "i,j,l,m,n,ko->i",
    "i,j,l,m,o,kn->i",
    "i,j,l,n,o,km->i",
    "i,k,l,m,n,jo->i",
    "i,k,l,m,o,jn->i",
    "i,k,l,n,o,jm->i",
    "j,k,l,m,n,io->i",
    "j,k,l,m,o,in->i",
    "j,k,l,n,o,im->i",
    "k,l,m,n,o,ij->i",
    "i,j,m,n,o,kl->i",
    "i,k,m,n,o,jl->i",
    "i,l,m,n,o,jk->i",
    "j,k,m,n,o,il->i",
    "j,l,m,n,o,ik->i",
]
all_elems = []
for i, string in enumerate(strings7_5_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(5):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            adders.append("[:-1]")
        elif (
            fullstrings[i][x] == "j" or fullstrings[i][x] == "k" or fullstrings[i][x] == "l" or fullstrings[i][x] == "o"
        ):
            adders.append("[1:]")

        else:
            adders.append("")
    extraadder = "["
    for x in range(5, 7):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            extraadder += ":-1,"
        elif (
            fullstrings[i][x] == "j" or fullstrings[i][x] == "k" or fullstrings[i][x] == "l" or fullstrings[i][x] == "o"
        ):
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
    replaced = (
        finalstring.replace("inp", "y")
        .replace("j", "k")
        .replace("mu", "x")
        .replace("l", "k")
        .replace("m", "l")
        .replace("n", "l")
        .replace("o", "l")
        .replace("p", "l")
        .replace("x", "mu")
        .replace("y", "inp")
    )
    finalstring = "x7_e_contracted_thirdV+=np.einsum(" + replaced
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
    newstring += "->i"
    # print("'%s',"%newstring)

strings7_3_upd = [
    "i,j,k,lmno->i",
    "i,j,l,kmno->i",
    "i,j,m,klno->i",
    "i,j,n,klmo->i",
    "i,j,o,klmn->i",
    "i,k,l,jmno->i",
    "i,k,m,jlno->i",
    "i,k,n,jlmo->i",
    "i,k,o,jlmn->i",
    "i,l,m,jkno->i",
    "i,l,n,jkmo->i",
    "i,l,o,jkmn->i",
    "i,m,n,jklo->i",
    "i,m,o,jkln->i",
    "i,n,o,jklm->i",
    "j,k,l,imno->i",
    "j,k,m,ilno->i",
    "j,k,n,ilmo->i",
    "j,k,o,ilmn->i",
    "j,l,m,ikno->i",
    "j,l,n,ikmo->i",
    "j,l,o,ikmn->i",
    "j,m,n,iklo->i",
    "j,m,o,ikln->i",
    "j,n,o,iklm->i",
    "k,l,m,ijno->i",
    "k,l,n,ijmo->i",
    "k,l,o,ijmn->i",
    "k,m,n,ijlo->i",
    "k,m,o,ijln->i",
    "k,n,o,ijlm->i",
    "l,m,n,ijko->i",
    "l,m,o,ijkn->i",
    "l,n,o,ijkm->i",
    "m,n,o,ijkl->i",
]
all_elems = []
for i, string in enumerate(strings7_3_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(3):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            adders.append("[:-1]")
        elif (
            fullstrings[i][x] == "j" or fullstrings[i][x] == "k" or fullstrings[i][x] == "l" or fullstrings[i][x] == "o"
        ):
            adders.append("[1:]")

        else:
            adders.append("")
    extraadder = "["
    for x in range(3, 7):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            extraadder += ":-1,"
        elif (
            fullstrings[i][x] == "j" or fullstrings[i][x] == "k" or fullstrings[i][x] == "l" or fullstrings[i][x] == "o"
        ):
            extraadder += "1:,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    returnstring = "'%s',mu" % string + adders[0]
    returnstring = (
        returnstring.replace("inp", "y")
        .replace("j", "k")
        .replace("mu", "x")
        .replace("l", "k")
        .replace("m", "l")
        .replace("n", "l")
        .replace("o", "l")
        .replace("p", "l")
        .replace("x", "mu")
        .replace("y", "inp")
    )
    finalstring = returnstring + ",mu" + adders[1] + ",mu" + adders[2] + ",fourth_expecs" + extraadder + ")"
    finalstring = "x7_e_contracted_thirdV+=np.einsum(" + finalstring
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
    newstring += "->i"
    # print("'%s',"%newstring)

strings7_1_upd = [
    "i,jklmno->i",
    "j,iklmno->i",
    "k,ijlmno->i",
    "l,ijkmno->i",
    "m,ijklno->i",
    "n,ijklmo->i",
    "o,ijklmn->i",
]

all_elems = []
for i, string in enumerate(strings7_1_upd[:]):
    # print(fullstrings[i])
    adders = []
    k_counter = 0
    l_counter = 0
    for x in range(1):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            adders.append("[:-1]")
        elif (
            fullstrings[i][x] == "j" or fullstrings[i][x] == "k" or fullstrings[i][x] == "l" or fullstrings[i][x] == "o"
        ):
            adders.append("[1:]")

        else:
            adders.append("")
    extraadder = "["
    for x in range(1, 7):
        if fullstrings[i][x] == "m" or fullstrings[i][x] == "n":
            extraadder += ":-1,"
        elif (
            fullstrings[i][x] == "j" or fullstrings[i][x] == "k" or fullstrings[i][x] == "l" or fullstrings[i][x] == "o"
        ):
            extraadder += "1:,"
        else:
            extraadder += ":,"
    extraadder = extraadder[:-1] + "]"
    returnstring = "'%s',mu" % string + adders[0]
    returnstring = (
        returnstring.replace("inp", "y")
        .replace("j", "k")
        .replace("mu", "x")
        .replace("l", "k")
        .replace("m", "l")
        .replace("n", "l")
        .replace("o", "l")
        .replace("p", "l")
        .replace("x", "mu")
        .replace("y", "inp")
    )
    finalstring = returnstring + ",sixth_expecs" + extraadder + ")"
    finalstring = "x7_e_contracted_thirdV+=np.einsum(" + finalstring
    all_elems.append(finalstring)
    # print(finalstring)
mylist = list(dict.fromkeys(all_elems))
for elem in mylist:
    count = countX(all_elems, elem)
    elem_splitted = elem.split("+=")
    print("%s+=%d*%s" % (elem_splitted[0], count, elem_splitted[1]))
