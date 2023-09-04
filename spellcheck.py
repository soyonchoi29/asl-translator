from spellchecker import SpellChecker

def spellCheck(text):
    spell = SpellChecker()
    text_array = text.split(" ")

    autocorrected = ""
    for word in text_array:
        if spell[word]:
            autocorrected += word + " "
        else:
            autocorrected += spell.correction(word)

    return autocorrected

# maybe want to build something that makes it more likely for o to be mispelled as c, s to a, etc.
# (similar hand shapes?)

# if __name__ == "__main__":
#
