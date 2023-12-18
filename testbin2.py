import math


def toBinary(a):
  l,m=[],[]
  for i in a:
    l.append(ord(i))
  for i in l:
    m.append(int(bin(i)[2:]))
  return m

#print("''Hello world'' in binary is ")
binv=toBinary("hi")
bb=''.join(str(binv))

lenb=len(binv)
i=0
bb=''
while i<lenb:
    bb+=str(binv[i])
    i+=1

#print(toBinary("hi"))
print(bb)
##################
def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m

print(toString(binv))
