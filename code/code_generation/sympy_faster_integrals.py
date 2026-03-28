from sympy import *

x=IndexedBase("x")
z=IndexedBase("z")
mu=IndexedBase("mu")
i,j,k,l,m,n,o,p=symbols("i,j,k,l,m,n,o,p",integer=True)
fifth=((z[i]+mu[i])*(z[j]+mu[j])*(z[k]+mu[k])*(z[l]+mu[l])*(z[m]+mu[m]))

eigth=((z[i]+mu[i])*(z[j]+mu[j])*(z[k]+mu[k])*(z[l]+mu[l])*(z[m]+mu[m])*(z[n]+mu[n])*(z[o]+mu[o])*(z[p]+mu[p]))
pow8=(z[i]+mu[i])**8
print(expand(pow8))
string=str(expand(eigth))
ledds=string.split("+")
two=[]
four=[]
six=[]
for ledd in ledds:
    if ledd.count("z")==2:
        two.append(ledd)
    elif ledd.count("z")==4:
        four.append(ledd)
    elif ledd.count("z")==6:
        six.append(ledd)
#print(two)
exprs_2=[]
exprs_4=[]
exprs_6=[]
for string in two:
    tempstring=string.strip().replace("mu","").replace("z","").replace("[","").replace("]","").replace("*","")
    exprs_2.append(tempstring[:6]+","+tempstring[6:]+"->ijklmnop")
print(exprs_2)
print(len(exprs_2))
for string in four:
    tempstring=string.strip().replace("mu","").replace("z","").replace("[","").replace("]","").replace("*","")
    exprs_4.append(tempstring[:4]+","+tempstring[4:]+"->ijklmnop")
print(exprs_4)
print(len(exprs_4))

for string in six:
    tempstring=string.strip().replace("mu","").replace("z","").replace("[","").replace("]","").replace("*","")
    exprs_6.append(tempstring[:2]+","+tempstring[2:]+"->ijklmnop")

print(exprs_6)
print(len(exprs_6))
