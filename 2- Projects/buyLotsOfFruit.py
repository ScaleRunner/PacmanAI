# Dennis Verheijden     4455770 KI
# Remco van der Heijden 4474139 KI


# buyLotsOfFruit.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
To run this script, type

  python buyLotsOfFruit.py
  
Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""
import sys
import re

fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75,
              'limes':0.75, 'strawberries':1.00}

def quickSort(lst):
    if len(lst) <= 1:
        return lst
    smaller = [x for x in lst[1:] if x < lst[0]]
    larger = [x for x in lst[1:] if x >= lst[0]]
    return quickSort(smaller) + [lst[0]] + quickSort(larger)

def buyLotsOfFruit(orderList):
    """
        orderList: List of (fruit, numPounds) tuples
            
    Returns cost of order
    """
    totalCost = 0.0
    for index in range (orderList.__len__()): #Voor elke waarde in de lijst
        if fruitPrices.__contains__(orderList[index][0]):
            totalCost += fruitPrices[orderList[index][0]] * orderList[index][1]
        else:
            print(orderList[index][0] + " is not in stock")
            return None
    return totalCost
    
# Main Method    
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [ ('apples', 2.0), ('pears', 3.0), ('limes', 4.0) ]
    print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))
    if len(sys.argv) == 1:
      args=input("Enter any command line arguments?")
    if args != '':
      sys.argv.extend(re.split(r' *',args))
    print(str(len(sys.argv)) + ': ')
    print(sys.argv)
