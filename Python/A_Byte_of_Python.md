>ref: [***A Byte of Python***](https://python.swaroopch.com/)

# Basics

## Comments
`# balabala`

## Strings
strings are immutable

strings in double quotes work exactly the same way as single quotes.
you can use both of them freely within the triple quotes

You specify the single quote as \' : 'What\'s your name?'.
Also, you have to indicate the backslash itself using the escape sequence \\.

To specify a multi-line string:
```python
# use a triple-quoted string
'''This is a multi-line string. This is the first line.
This is the second line.
"What's your name?," I asked.
He said "Bond, James Bond."
'''
# use an escape sequence for the newline character - \n
'This is the first line\nThis is the second line'
```

specify a raw string: by prefixing r or R to the string.
```python
r"Newlines are indicated by \n"
```

## The format method
```python
age = 20
name = 'Swaroop'
print('{0} was {1} years old when he wrote this book'.format(name, age))
print('Why is {0} playing with that python?'.format(name))

# Also note that the numbers are optional
age = 20
name = 'Swaroop'
print('{} was {} years old when he wrote this book'.format(name, age))
print('Why is {} playing with that python?'.format(name))
```
## Data Types
The basic types are numbers and strings, create own types using classes

## Object
Python is strongly object-oriented in the sense that everything is an object including numbers, strings and functions.

# Operators and Expressions
`** (power)`

  Returns x to the power of y
  3 ** 4 gives 81 (i.e. 3 * 3 * 3 * 3)

`// (divide and floor)`

  Divide x by y and round the answer down to the nearest whole number
  13 // 3 gives 4
  -13 // 3 gives -5

`& (bit-wise AND)`

`| (bit-wise OR)`

`^ (bit-wise XOR)`

`~ (bit-wise invert)`

`not (boolean NOT)`

`and (boolean AND)`

`or (boolean OR)`

# Control flow

## if...elif...else & while & break & continue
```python
while True:
  if guess == number:
      # New block here
      break
  elif guess < number:
      # Another block
      continue
  else:
      # you must have guessed > number to reach here
```
Note there is **no** switch statement in Python.

## for...in
```python
for i in range(1, 5):
    print(i)  # print 1 to 4
else:
    print('The for loop is over')
```
>range(1,5) gives the sequence [1, 2, 3, 4]. By default, range takes a step count of 1. If we supply a third number to range, then that becomes the step count. For example, range(1,5,2) gives [1,3]. Remember that the range extends up to the second number 
>
>i.e. it does **not** include the second number.
>
>Note that range() generates only one number at a time, if you want the full list of numbers, call list() on the range(), for example, list(range(5)) will result in [0, 1, 2, 3, 4].
>
>for i in range(1,5) is equivalent to for i in [1, 2, 3, 4] which is like assigning each number (or object) in the sequence to i, one at a time

# Function

## Functions & Params & Default Argument Values & VarArgs parameters & `return` & DocStrings
```python
def func(a, b=1, c=2):
  '''documentation strings here.
  balabala'''
    # block belonging to the function
    print('a is', a, 'and b is', b, 'and c is', c)
    return a+b
# End of function

func(x, y)  # call the function
print(func.__doc__)  # access the docstring

def total(a=5, *numbers, **phonebook):
  #iterate through all the items in tuple
  for single_item in numbers:
    print('single_item', single_item)

  #iterate through all the items in dictionary    
  for first_part, second_part in phonebook.items():
    print(first_part,second_part)
    
print(total(10,1,2,3,Jack=1123,John=2231,Inge=1560))
```

## Local Variables
```python
x = 50

def func(x):
  print('x is', x)
  x = 2
  print('Changed local x to', x)
    
func(x)
print('x is still', x)
```

## The `global` statement
If you want to assign a value to a name defined at the top level of the program (i.e. not inside any kind of scope such as functions or classes), we do this using the global statement.

```python
x = 50

def func():
  global x
  print('x is', x)
  x = 2
  print('Changed global x to', x)

func()
print('Value of x is', x)
```
>The global statement is used to declare that x is a global variable - hence, when we assign a value to x inside the function, that change is reflected when we use the value of x in the main block.
>
>You can use the values of such variables defined outside the function (assuming there is no variable with the same name within the function). However, this is not encouraged and should be avoided since it becomes unclear to the reader of the program as to where that variable's definition is. Using the global statement makes it amply clear that the variable is defined in an outermost block.

# Modules
```python
import sys

print('The command line arguments are:')
for i in sys.argv:
    print(i)

print('\n\nThe PYTHONPATH is', sys.path, '\n')
```
>Here, when we execute `python module_using_sys.py we are arguments`, the name of the script running is always the first element in the sys.argv list. So, in this case we will have 'module_using_sys.py' as sys.argv[0], 'we' as sys.argv[1], 'are' as sys.argv[2] and 'arguments' as sys.argv[3]. Notice that Python starts counting from 0 and not 1.
>
>The `sys.path` contains the list of directory names where modules are imported from. Observe that the first string in sys.path is empty - this empty string indicates that the current directory is also part of the sys.path which is same as the PYTHONPATH environment variable.

## Byte-compiled .pyc files
py source code is first compiled to byte code as .pyc. This byte code can be interpreted (official CPython), or JIT compiled (PyPy). Python source code (.py) can be compiled to different byte code also like IronPython (.Net) or Jython (JVM). There are multiple implementations of Python language.

## The `from..import` statement
```python
from sys import argv

# This will only import all public names
# but not private ones start with double underscores
from mymodule import *
```
>In general, **avoid** using the from..import statement, use the import statement instead. This is because your program will avoid name clashes and will be more readable.
>
>Remember that you should **avoid** using import-star, i.e. `from mymodule import *`. One of Python's guiding principles is that "Explicit is better than Implicit".

## A module's `__name__`
Every Python module has its `__name__` defined. If this is `__main__`, that implies that the module is being run standalone by the user. When a module is imported for the first time, the code it contains gets executed. We can use this to make the module behave in different ways depending on whether it is being used by itself or being imported from another module.
```python
if __name__ == '__main__':
    print('This program is being run by itself')
else:
    print('I am being imported from another module')
```

## Making Your Own Modules
Every Python program is also a module. You just have to make sure it has a .py extension, there is nothing particularly special about it compared to our usual Python program.

Remember that the module should be placed either in the same directory as the program from which we import it, or in one of the directories listed in sys.path.

## The `dir` function
Built-in `dir()` function returns list of names defined by an object. If the object is a module, this list includes functions, classes and variables, defined inside that module.

A note on `del` - this statement is used to delete a variable/name and after the statement has run. There is also a vars() function which can potentially give you the attributes and their values.

# Data Structures
There are four built-in data structures in Python:
`list`, `tuple`, `dictionary` and `set`.

## List
A list is a data structure that holds an ordered collection of items i.e. you can store a sequence of items in a list.Once you have created a list, you can add, remove or search for items in the list. Since we can add and remove items, we say that a list is a mutable data type.
```python
# This is my shopping list
shoplist = ['apple', 'mango', 'carrot', 'banana']

print('I have', len(shoplist), 'items to purchase.')
print('These items are:', end=' ')
for item in shoplist:
    print(item, end=' ')

print('\nI also have to buy rice.')
shoplist.append('rice')
print('My shopping list is now', shoplist)

print('I will sort my list now')
shoplist.sort()
print('Sorted shopping list is', shoplist)

print('The first item I will buy is', shoplist[0])
olditem = shoplist[0]
del shoplist[0]
print('I bought the', olditem)
print('My shopping list is now', shoplist)
```

## Tuple
Tuples are used to hold together multiple objects. Think of them as similar to lists, but without the extensive functionality that the list class gives you. One major feature of tuples is that they are immutable like strings.
```python
# I would recommend always using parentheses
# to indicate start and end of tuple
# even though parentheses are optional.
# Explicit is better than implicit.
zoo = ('python', 'elephant', 'penguin')
print('Number of animals in the zoo is', len(zoo))

new_zoo = 'monkey', 'camel', zoo
print('Number of cages in the new zoo is', len(new_zoo))
print('All animals in new zoo are', new_zoo)
print('Animals brought from old zoo are', new_zoo[2])
print('Last animal brought from old zoo is', new_zoo[2][2])
print('Number of animals in the new zoo is',
      len(new_zoo)-1+len(new_zoo[2]))


$ python ds_using_tuple.py
Number of animals in the zoo is 3
Number of cages in the new zoo is 3
All animals in new zoo are ('monkey', 'camel', ('python', 'elephant', 'penguin'))
Animals brought from old zoo are ('python', 'elephant', 'penguin')
Last animal brought from old zoo is penguin
Number of animals in the new zoo is 5
```

>Tuple with 0 or 1 items
>
>An empty tuple is constructed by an empty pair of parentheses such as myempty = (). However, a tuple with a single item is not so simple. You have to specify it using a comma following the first (and only) item so that Python can differentiate between a tuple and a pair of parentheses surrounding the object in an expression i.e. you have to specify singleton = (2 , ) if you mean you want a tuple containing the item 2.

## Dictionary
> just like object in javascript (key-value pairs)

Note that you can use only immutable objects (like strings) for the keys of a dictionary but you can use either immutable or mutable objects for the values of the dictionary.

Pairs of keys and values are specified in a dictionary by using the notation d = {key1 : value1, key2 : value2 }.

```python

ab = {
    'Swaroop': 'swaroop@swaroopch.com',
    'Larry': 'larry@wall.org',
    'Matsumoto': 'matz@ruby-lang.org',
    'Spammer': 'spammer@hotmail.com'
}

print("Swaroop's address is", ab['Swaroop'])

# Deleting a key-value pair
del ab['Spammer']

print('\nThere are {} contacts in the address-book\n'.format(len(ab)))

for name, address in ab.items():
    print('Contact {} at {}'.format(name, address))

# Adding a key-value pair
ab['Guido'] = 'guido@python.org'

if 'Guido' in ab:
    print("\nGuido's address is", ab['Guido'])

```

## Sequence

Lists, tuples and strings are examples of sequences. The major features are membership tests, (i.e. the in and not in expressions) and indexing operations, which allow us to fetch a particular item in the sequence directly.

```python
shoplist = ['apple', 'mango', 'carrot', 'banana']
name = 'swaroop'

# Indexing or 'Subscription' operation #
print('Item 0 is', shoplist[0])
print('Item 1 is', shoplist[1])
print('Item 2 is', shoplist[2])
print('Item 3 is', shoplist[3])
print('Item -1 is', shoplist[-1])
print('Item -2 is', shoplist[-2])
print('Character 0 is', name[0])

# Slicing on a list #
print('Item 1 to 3 is', shoplist[1:3])
print('Item 2 to end is', shoplist[2:])
print('Item 1 to -1 is', shoplist[1:-1])
print('Item start to end is', shoplist[:])

# Slicing on a string #
print('characters 1 to 3 is', name[1:3])
print('characters 2 to end is', name[2:])
print('characters 1 to -1 is', name[1:-1])
print('characters start to end is', name[:])

# You can also provide a third argument for the slice, which is the step for the slicing (by default, the step size is 1):
>>> shoplist = ['apple', 'mango', 'carrot', 'banana']
>>> shoplist[::1]
['apple', 'mango', 'carrot', 'banana']
>>> shoplist[::2]
['apple', 'carrot']
>>> shoplist[::3]
['apple', 'banana']
>>> shoplist[::-1]
['banana', 'carrot', 'mango', 'apple']
```

## Set
Sets are unordered collections of simple objects. These are used when the existence of an object in a collection is more important than the order or how many times it occurs.

```python
>>> bri = set(['brazil', 'russia', 'india'])
>>> 'india' in bri
True
>>> 'usa' in bri
False
>>> bric = bri.copy()
>>> bric.add('china')
>>> bric.issuperset(bri)
True
>>> bri.remove('russia')
>>> bri & bric # OR bri.intersection(bric)
{'brazil', 'india'}
```

## References
variable only refers to the object and does not represent the object itself.

```python
print('Simple Assignment')
shoplist = ['apple', 'mango', 'carrot', 'banana']
# mylist is just another name pointing to the same object!
mylist = shoplist

# I purchased the first item, so I remove it from the list
del shoplist[0]

print('shoplist is', shoplist)
print('mylist is', mylist)
# Notice that both shoplist and mylist both print
# the same list without the 'apple' confirming that
# they point to the same object

print('Copy by making a full slice')
# Make a copy by doing a full slice
mylist = shoplist[:]
# Remove first item
del mylist[0]

print('shoplist is', shoplist)
print('mylist is', mylist)
# Notice that now the two lists are different
```

## More About Strings
The strings that you use in program are all objects of the class str. Strings are also objects and have methods which do everything from checking part of a string to stripping spaces.

```python
# This is a string object
name = 'Swaroop'

if name.startswith('Swa'):
    print('Yes, the string starts with "Swa"')

if 'a' in name:
    print('Yes, it contains the string "a"')

if name.find('war') != -1:
    print('Yes, it contains the string "war"')

delimiter = '_*_'
mylist = ['Brazil', 'Russia', 'India', 'China']
print(delimiter.join(mylist))
```

# Object Oriented Programming
Objects can store data using ordinary variables that belong to the object. Variables that belong to an object or class are referred to as **fields**. Objects can also have functionality by using functions that belong to a class. Such functions are called **methods** of the class. This terminology is important because it helps us to differentiate between functions and variables which are independent and those which belong to a class or object. Collectively, the fields and methods can be referred to as the **attributes** of that class.

>Note that even integers are treated as objects (of the int class). This is unlike C++ and Java (before version 1.5) where integers are primitive native types.
>See help(int) for more details on the class.
>C# and Java 1.5 programmers will find this similar to the boxing and unboxing concept.

## The `self`
Class methods have only one specific difference from ordinary functions - they must have an extra first name that has to be added to the beginning of the parameter list, but you **do not** give a value for this parameter when you call the method, Python will provide it. This particular variable refers to the object itself, and by convention, it is given the name `self`.

>The self in Python is equivalent to the this pointer in C++ and the this reference in Java and C#.

## Classes
```python
# the simplest class
class Person:
    pass  # An empty block

p = Person()
print(p) # <__main__.Person instance at 0x10171f518>

# methods
class Person:
    def say_hi(self):
        print('Hello, how are you?')

Person().say_hi()

# The __init__ method
class Person:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print('Hello, my name is', self.name)

p = Person('Swaroop')
p.say_hi()
```

## Class And Object Variables
**Class variables** are shared - they can be accessed by all instances of that class.

**Object variables** are owned by each individual object/instance of the class.

```python
class Robot:
    """Represents a robot, with a name."""

    # A class variable, counting the number of robots
    population = 0

    def __init__(self, name):
        """Initializes the data."""
        self.name = name
        print("(Initializing {})".format(self.name))

        # When this person is created, the robot
        # adds to the population
        Robot.population += 1

    def die(self):
        """I am dying."""
        print("{} is being destroyed!".format(self.name))

        Robot.population -= 1

        if Robot.population == 0:
            print("{} was the last one.".format(self.name))
        else:
            print("There are still {:d} robots working.".format(
                Robot.population))

    def say_hi(self):
        """Greeting by the robot.

        Yeah, they can do that."""
        print("Greetings, my masters call me {}.".format(self.name))

    @classmethod
    def how_many(cls):
        """Prints the current population."""
        print("We have {:d} robots.".format(cls.population))


droid1 = Robot("R2-D2")
droid1.say_hi()
Robot.how_many()

droid2 = Robot("C-3PO")
droid2.say_hi()
Robot.how_many()

print("\nRobots can do some work here.\n")

print("Robots have finished their work. So let's destroy them.")
droid1.die()
droid2.die()

Robot.how_many()
```

## Inheritance
One of the major benefits of object oriented programming is reuse of code and one of the ways this is achieved is through the inheritance mechanism.

>A note on terminology - if more than one class is listed in the inheritance tuple, then it is called multiple inheritance.

```python
class SchoolMember:
    '''Represents any school member.'''
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('(Initialized SchoolMember: {})'.format(self.name))

    def tell(self):
        '''Tell my details.'''
        print('Name:"{}" Age:"{}"'.format(self.name, self.age), end=" ")


class Teacher(SchoolMember):
    '''Represents a teacher.'''
    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: "{:d}"'.format(self.salary))


class Student(SchoolMember):
    '''Represents a student.'''
    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Marks: "{:d}"'.format(self.marks))

t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)

# prints a blank line
print()

members = [t, s]
for member in members:
    # Works for both Teachers and Students
    member.tell()

```

## Input and Output
For example, you would want to take input from the user and then print some results back. We can achieve this using the `input()` function and `print` function respectively.

##Input from user
```python
def reverse(text):
    return text[::-1]

def is_palindrome(text):
    return text == reverse(text)

something = input("Enter text: ")
if is_palindrome(something):
    print("Yes, it is a palindrome")
else:
    print("No, it is not a palindrome")

```

## Files
You can open and use files for reading or writing by creating an object of the `file` class and using its `read`, `readline` or `write` methods appropriately to read from or write to the file. Then finally, when you are finished with the file, you call the `close` method to tell Python that we are done using the file.

```python
poem = '''\
Programming is fun
When the work is done
if you wanna make your work also fun:
    use Python!
'''

# Open for 'w'riting
f = open('poem.txt', 'w')
# Write text to file
f.write(poem)
# Close the file
f.close()

# If no mode is specified,
# 'r'ead mode is assumed by default
f = open('poem.txt')
while True:
    line = f.readline()
    # Zero length indicates EOF
    if len(line) == 0:
        break
    # The `line` already has a newline
    # at the end of each line
    # since it is reading from a file.
    print(line, end='')
# close the file
f.close()
```

## Pickle
Python provides a standard module called `pickle` using which you can store any plain Python object in a file and then get it back later. This is called storing the object ***persistently***.

```python
import pickle

# The name of the file where we will store the object
shoplistfile = 'shoplist.data'
# The list of things to buy
shoplist = ['apple', 'mango', 'carrot']

# Write to the file
f = open(shoplistfile, 'wb')
# Dump the object to a file
pickle.dump(shoplist, f)
f.close()

# Destroy the shoplist variable
del shoplist

# Read back from the storage
f = open(shoplistfile, 'rb')
# Load the object from the file
storedlist = pickle.load(f)
print(storedlist)
```

## Unicode
When we read or write to a file or when we talk to other computers on the Internet, we need to convert our unicode strings into a format that can be sent and received, and that format is called "UTF-8". We can read and write in that format, using a simple keyword argument to our standard open function:

```python
# encoding=utf-8
import io

f = io.open("abc.txt", "wt", encoding="utf-8")
f.write(u"Imagine non-English language here")
f.close()

text = io.open("abc.txt", encoding="utf-8").read()
print(text)
```

>NOTE: If you are using Python 2, and we want to be able to read and write other non-English languages, we need to use the unicode type, and it all starts with the character u, e.g. u"hello world"

# Exceptions

## `try...except...else...`

```python
class ShortInputException(Exception):
    '''A user-defined exception class.'''
    def __init__(self, length, atleast):
        Exception.__init__(self)
        self.length = length
        self.atleast = atleast

try:
    text = input('Enter something --> ')
    if len(text) < 3:
        raise ShortInputException(len(text), 3)
    # Other work can continue as usual here
except EOFError:
    print('Why did you do an EOF on me?')
except ShortInputException as ex:
    print(('ShortInputException: The input was ' +
           '{0} long, expected at least {1}')
          .format(ex.length, ex.atleast))
else:
    print('No exception was raised.')
```

## `try...except...finally...`

```python
import sys
import time

f = None
try:
    f = open("poem.txt")
    # Our usual file-reading idiom
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        print(line, end='')
        sys.stdout.flush()
        print("Press ctrl+c now")
        # To make sure it runs for a while
        time.sleep(2)
except IOError:
    print("Could not find file poem.txt")
except KeyboardInterrupt:
    print("!! You cancelled the reading from the file.")
finally:
    if f:
        f.close()
    print("(Cleaning up: Closed the file)")
```

## The `with` statement

```python
with open("poem.txt") as f:
    for line in f:
        print(line, end='')
```

The output should be same as the previous example. The difference here is that we are using the open function with the with statement - we leave the closing of the file to be done automatically by with open.

What happens behind the scenes is that there is a protocol used by the `with` statement. It fetches the object returned by the `open` statement, let's call it "thefile" in this case.

It always calls the `thefile.__enter__` function before starting the block of code under it and always calls `thefile.__exit__` after finishing the block of code.

So the code that we would have written in a `finally` block should be taken care of automatically by the `__exit__` method. This is what helps us to avoid having to use explicit try..finally statements repeatedly.


# Standard Library

[The Python Standard Library](https://docs.python.org/3/library/index.html)

[Python 3 Module of the Week](https://pymotw.com/3/index.html)

# More

## Passing tuples around

```python
# return two different values from a function
>>> def get_error_details():
...     return (2, 'details')
...
>>> errnum, errstr = get_error_details()
>>> errnum
2
>>> errstr
'details'

# the fastest way to swap two variables
>>> a = 5; b = 8
>>> a, b
(5, 8)
>>> a, b = b, a
>>> a, b
(8, 5)
```

## Special Methods
Special methods are used to mimic certain behaviors of built-in types.

- `__init__(self, ...)`
This method is called just before the newly created object is returned for usage.

- `__del__(self)`
Called just before the object is destroyed (which has unpredictable timing, so avoid using this)

- `__str__(self)`
Called when we use the print function or when str() is used.

- `__lt__(self, other)`
Called when the less than operator (<) is used. Similarly, there are special methods for all the operators (+, >, etc.)

- `__getitem__(self, key)`
Called when x[key] indexing operation is used.

- `__len__(self)`
Called when the built-in len() function is used for the sequence object.

[see all](https://docs.python.org/3/reference/datamodel.html#special-method-names)

## Lambda Forms

```python
# sort method of a list can take a `key` parameter which determines how the list is sorted
points = [{'x': 2, 'y': 3},
          {'x': 4, 'y': 1}]
points.sort(key=lambda i: i['y'])
print(points)
```

## List Comprehension

```python
listone = [2, 3, 4]
listtwo = [2*i for i in listone if i > 2]
print(listtwo) # [6, 8]
```

## Receiving Tuples and Dictionaries in Functions

```python
# There is a special way of receiving parameters to a function as a tuple or a dictionary(key/value pairs) using the * or ** prefix respectively.
>>> def powersum(power, *args):
...     '''Return the sum of each argument raised to the specified power.'''
...     total = 0
...     for i in args:
...         total += pow(i, power)
...     return total
...
>>> powersum(2, 3, 4)
25
>>> powersum(2, 10)
100
```

## The assert statement

```python
>>> mylist = ['item']
>>> assert len(mylist) >= 1
>>> mylist.pop()
'item'
>>> assert len(mylist) >= 1
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError
```

## Decorators
Decorators are a shortcut to applying wrapper functions. This is helpful to "wrap" functionality with the same code over and over again.

```python
from time import sleep
from functools import wraps
import logging
logging.basicConfig()
log = logging.getLogger("retry")


def retry(f):
    @wraps(f)
    def wrapped_f(*args, **kwargs):
        MAX_ATTEMPTS = 5
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return f(*args, **kwargs)
            except:
                log.exception("Attempt %s/%s failed : %s",
                              attempt,
                              MAX_ATTEMPTS,
                              (args, kwargs))
                sleep(10 * attempt)
        log.critical("All %s attempts failed : %s",
                     MAX_ATTEMPTS,
                     (args, kwargs))
    return wrapped_f


counter = 0


@retry
def save_to_database(arg):
    print("Write to a database or make a network call or etc.")
    print("This will be automatically retried if exception is thrown.")
    global counter
    counter += 1
    # This will throw an exception in the first call
    # And will work fine in the second call (i.e. a retry)
    if counter < 2:
        raise ValueError(arg)


if __name__ == '__main__':
    save_to_database("Some bad value")
```