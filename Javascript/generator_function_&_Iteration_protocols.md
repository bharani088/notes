# Generator

The Generator object is returned by a generator function and it conforms to both the iterable protocol and the iterator protocol.

## Methods

Generator.prototype.next()
Returns a value yielded by the yield expression.
Generator.prototype.return()
Returns the given value and finishes the generator.
Generator.prototype.throw()
Throws an error to a generator.

## function*

The function* declaration (function keyword followed by an asterisk) defines a generator function, which returns a Generator object.

You can also define generator functions using the GeneratorFunction constructor and a function* expression.

Generators are functions which can be exited and later re-entered. Their context (variable bindings) will be saved across re-entrances.

Calling a generator function does not execute its body immediately; an iterator object for the function is returned instead. When the iterator's next() method is called, the generator function's body is executed until the first yieldexpression, which specifies the value to be returned from the iterator or, with `yield*`, delegates to another generator function. The next() method returns an object with a value property containing the yielded value and a done property which indicates whether the generator has yielded its last value as a boolean. Calling the next() method with an argument will resume the generator function execution, replacing the yield statement where execution was paused with the argument from next(). 

A return statement in a generator, when executed, will make the generator done. If a value is returned, it will be passed back as the value. A generator which has returned will not yield any more values.



## yield

The yield keyword is used to pause and resume a generator function (function* or legacy generator function).
[rv] = yield [expression];
expression
Defines the value to return from the generator function via the iterator protocol. If omitted, undefined is returned instead.
rv
Returns the optional value passed to the generator's next() method to resume its execution.

## yield*

The yield* expression is used to delegate to another generator or iterable object.
yield* [[expression]];
expression
The expression which returns an iterable object.


# Iteration protocols

## The iterable protocol

The iterable protocol allows JavaScript objects to define or customize their iteration behavior, such as what values are looped over in a for..of construct. Some built-in types are built-in iterables with a default iteration behavior, such as Array or Map, while other types (such as Object) are not.

In order to be iterable, an object must implement the @@iterator method, meaning that the object (or one of the objects up its prototype chain) must have a property with a @@iterator key which is available via constant Symbol.iterator:

Property
Value
[Symbol.iterator]
A zero arguments function that returns an object, conforming to the iterator protocol. 
Whenever an object needs to be iterated (such as at the beginning of a for..of loop), its @@iterator method is called with no arguments, and the returned iterator is used to obtain the values to be iterated.

## The iterator protocol

The iterator protocol defines a standard way to produce a sequence of values (either finite or infinite).

An object is an iterator when it implements a next() method with the following semantics:

Property
Value
next
A zero arguments function that returns an object with two properties:

* done (boolean)
    * Has the value true if the iterator is past the end of the iterated sequence. In this case valueoptionally specifies the return value of the iterator. The return values are explained here.
    * Has the value false if the iterator was able to produce the next value in the sequence. This is equivalent of not specifying the done property altogether.
* value - any JavaScript value returned by the iterator. Can be omitted when done is true.
The next method always has to return an object with appropriate properties including done and value. If a non-object value gets returned (such as false or undefined), a TypeError ("iterator.next() returned a non-object value") will be thrown.


## A generator object is both, iterator and iterable

var aGeneratorObject = function* () {
    yield 1;
    yield 2;
    yield 3;
}();
typeof aGeneratorObject.next;
// "function", because it has a next method, so it's an iterator
typeof aGeneratorObject[Symbol.iterator];
// "function", because it has an @@iterator method, so it's an iterable
aGeneratorObject[Symbol.iterator]() === aGeneratorObject;
// true, because its @@iterator method return its self (an iterator), so it's an well-formed iterable
[...aGeneratorObject];
// [1, 2, 3]


## Iterable examples

### Built-in iterables

String, Array, TypedArray, Map and Set are all built-in iterables, because each of their prototype objects implements an @@iterator method.

### User-defined iterables

We can make our own iterables like this:

var myIterable = {};
myIterable[Symbol.iterator] = function* () {
    yield 1;
    yield 2;
    yield 3;
};
[...myIterable]; // [1, 2, 3]

### Built-in APIs accepting iterables

There are many APIs that accept iterables, for example: Map([iterable]), WeakMap([iterable]), Set([iterable]) and WeakSet([iterable]):

var myObj = {};
new Map([[1, 'a'], [2, 'b'], [3, 'c']]).get(2);               // "b"
new WeakMap([[{}, 'a'], [myObj, 'b'], [{}, 'c']]).get(myObj); // "b"
new Set([1, 2, 3]).has(3);                               // true
new Set('123').has('2');                                 // true
new WeakSet(function* () {
    yield {};
    yield myObj;
    yield {};
}()).has(myObj);                                         // true
Also see Promise.all(iterable), Promise.race(iterable), and Array.from().

### Syntaxes expecting iterables

Some statements and expressions expect iterables, for example the for-of loops, spread operator, yield*, and destructuring assignment:

for(let value of ['a', 'b', 'c']){
    console.log(value);
}
// "a"
// "b"
// "c"

[...'abc']; // ["a", "b", "c"]

function* gen() {
  yield* ['a', 'b', 'c'];
}

gen().next(); // { value:"a", done:false }

[a, b, c] = new Set(['a', 'b', 'c']);
a // “a"

