import enchant

d = enchant.Dict("en_US")

def check(word):
    return d.check(word)

def suggestions(word):
    return d.suggest(word)

if __name__ == "__main__":
    print check("hec")
    print suggestions("sudip")