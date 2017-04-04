
# def addQuestion(que, ans):
#     with open('ans.txt','a+') as f:
#         f.write(ans+"\n")
# addQuestion("q", "test")

f=open('database.txt','r')
test=f.readlines()
test = [q.strip().replace('?', '') for q in test]

print(test)