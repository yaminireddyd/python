file1 = input("enter file path:")
word_count = {}
file = open(file1 , 'r') #open the file
readfile = file.read().lower() #reading each and every line
des = readfile.strip("!()-[]{};:'\,<>./?@#$%^&*_~ ")
desired=des.split() # splitting without spaces
print (desired)
for i in desired:
    count = word_count.get(i,0)          #counting each word and storing in word_count
    word_count[i] = count+1
frequency_count = word_count.keys()      #creating list of words and count
for words in frequency_count:
    print(words , word_count[words])
