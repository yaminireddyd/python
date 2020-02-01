n=int(input("enter number of students"))
lis=[]
for i in range(n):
    ele=int(input("enter lbs"))
    lis.append(round(float(ele)*0.454,2))
print("final values after converting into kgs")
print(lis)