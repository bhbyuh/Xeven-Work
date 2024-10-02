#Task1

#ascending order sorting
list=[-1,78,-89,12,10]

for i in range(len(list)):
    for j in range(len(list)-i-1):
        if(list[j]>list[j+1]):
            temp=list[j]
            list[j]=list[j+1]
            list[j+1]=temp

#Task2
print("Original List:" ,list)
print("Sorted List:" ,sorted(list))

print("Original List:" ,list)

list.sort()
print("Sort List:" ,list)

#Task3
#--init--
class product:
    def __init__(self,a,b):
        self.a=a
        self.b=b
        print("Initialize")
    
    def __call__(self, num1,num2,num3):
        return self.a*num1+self.b*num2+num3

p1=product(2,3)

print(p1(1,2,4))
#Task4
lambda_func=lambda a,b:a+b 

print(lambda_func(1,2))

max=lambda a,b: a if(a>b) else b

print(max(1, 2))