class Account:
    def __init__(self,balance,name):
        self.balance=balance
        self.name=name
    
    def deposit(self,amount):
        self.balance=self.balance+amount
        print("Amount successfully added")
    
    def withdraw(self,amount):
        if(self.balance-amount<0):
            return "You have no Much Amount"
        else:
            self.balance=self.balance-amount
            return amount
    
    def check_balance(self):
        print("Your balance is ",self.balance)
    
p1=Account(12000,'Ali')

p1.check_balance()
p1.deposit(12)
p1.check_balance()

print(p1.withdraw(1200000))