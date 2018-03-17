# Arrows and Lexical This

Arrows are a function shorthand using the => syntax. They are syntactically similar to the related feature in C#, Java 8 and CoffeeScript. They support both expression and statement bodies. Unlike functions, arrows share the same lexical this as their surrounding code.
箭头函数就是个简写形式的函数表达式，并且它拥有词法作用域的this值（即不会新产生自己作用域下的this, arguments, super 和 new.target 等对象）。此外，箭头函数总是匿名的。

#Classes

ES2015 classes are a simple sugar over the prototype-based OO pattern. Having a single convenient declarative form makes class patterns easier to use, and encourages interoperability. Classes support prototype-based inheritance, super calls, instance and static methods and constructors.
ES6 中的类实际上就是个函数
类声明和函数声明不同的一点是，函数声明存在变量提升现象，而类声明不会。也就是说，你必须先声明类，然后才能使用它，否则代码会抛出 ReferenceError 异常
The static keyword defines a static method for a class. Static methods are called without instantiating their class and are also not callable when the class is instantiated. 

#Enhanced Object Literals

Object literals are extended to support setting the prototype at construction, shorthand for foo: foo assignments, defining methods and making super calls

# Template Strings

# Destructuring

Destructuring allows binding using pattern matching, with support for matching arrays and objects. Destructuring is fail-soft, similar to standard object lookup foo["bar"], producing undefined values when not found.

# Default + Rest + Spread

Callee-evaluated default parameter values. Turn an array into consecutive arguments in a function call. Bind trailing parameters to an array. Rest replaces the need for arguments and addresses common cases more directly.

# Let + Const

Block-scoped binding constructs. let is the new var. const is single-assignment. Static restrictions prevent use before assignment.

# Iterators + For..Of
see [Iteration protocols](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Iteration_protocols#iterator)

# Generators

Generators simplify iterator-authoring using function* and yield. A function declared as function* returns a Generator instance. 

# Unicode

# Modules

Language-level support for modules for component definition

# Map + Set + WeakMap + WeakSet

# Proxies
see Proxy

# Symbols

A symbol is a unique and immutable data type and may be used as an identifier for object properties. The symbol object is an implicit object wrapper for the symbol primitive data type.
 Symbols are unique (like gensym), but not private since they are exposed via reflection features like Object.getOwnPropertySymbols.

# Subclassable Built-ins

# Math + Number + String + Object APIs

Many new library additions, including core Math libraries, Array conversion helpers, and Object.assign for copying.

# Binary and Octal Literals

# Promises

# Reflect API

Full reflection API exposing the runtime-level meta-operations on objects.

# Tail Calls