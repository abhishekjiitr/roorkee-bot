
# def addQuestion(que, ans):
#     with open('ans.txt','a+') as f:
#         f.write(ans+"\n")
# addQuestion("q", "test")

f=open('database.txt','r')
testing=f.readlines()
testing = [q.strip().replace('?', '') for q in test]

print(testing)
