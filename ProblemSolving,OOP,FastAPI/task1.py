with open("file.txt",'r') as f:
    data=f.read()
print(data)
text=''

for line in data:
    text=text+line

frequency_dict={}

text=text.replace('\n','')

for character in text:    
    [character]=text.count(character)

print(frequency_dict)
