__author__ = 'Dennis'

def inverse(getal): #declaratie voor de functie
    return 1/getal

l = [1,2,3]
print (min(l, key=inverse)) #pak het kleinste element waarvoor de functie geldt
