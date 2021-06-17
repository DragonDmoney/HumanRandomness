f=open("../inputMe.txt")
w=open("output.txt","w")
numbers = []

for i in f:
    for n in i:
        numbers.append(n)

totalNumbers = len(numbers)-1
w.write("totalNumbers: "+str(totalNumbers)+"\n")
print(numbers)
for i in range(9):
    i+=1
    print(i,numbers.count(str(i)),numbers.count(str(i))/totalNumbers*100)
    w.write(str(i)+"   "+str(numbers.count(str(i)))+"   "+str(numbers.count(str(i))/totalNumbers*100)+"\n")
