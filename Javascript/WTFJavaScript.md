> [https://github.com/denysdovhan/wtfjs](https://github.com/denysdovhan/wtfjs)

Read more about specifications:

* [**12.8.3** The Addition Operator (`+`)](https://www.ecma-international.org/ecma-262/#sec-addition-operator-plus)
* [**7.1.1** ToPrimitive(`input` [,`PreferredType`])](https://www.ecma-international.org/ecma-262/#sec-toprimitive)
* [**7.1.12** ToString(`argument`)](https://www.ecma-international.org/ecma-262/#sec-tostring)

## `[]` is truthy, but not `true`
```js
!![]       // -> true
[] == true // -> false
```
```js
[] == ![] // -> true
```
![] evaluates to false because the reference is truthy. [] can be converted to an number (0 in this case) which is falsey.

## `null` is falsy, but not `false`
Despite the fact that `null` is falsy value, it's not equal to `false`.
```js
!!null        // -> false
null == false // -> false
```
At the same time, other falsy values, like `0` or `''` are equal to `false`.
```js
0 == false  // -> true
'' == false // -> true
```

## fooNaN
```js
"foo" + + "bar" // -> 'fooNaN'
```
The expression is evaluted as `'foo' + (+'bar')`, which converts `'bar'` to not a number.
```js
NaN === NaN // -> false
```
typeof(NaN) // -> ’number’
The specification strictly defines the logic behind this behavior:
> 1. If `Type(x)` is different from `Type(y)`, return **false**.
> 2. If `Type(x)` is Number, then
>     1. If `x` is **NaN**, return **false**.
>     2. If `y` is **NaN**, return **false**.
> Four mutually exclusive relations are possible: less than, equal, greater than, and unordered. The last case arises when at least one operand is NaN. Every NaN shall compare unordered with everything, including itself.

## Minimal value is greater than zero
```js
Number.MIN_VALUE > 0 // -> true
```
>Number.MIN_VALUE is 5e-324, i.e. the smallest positive number that can be represented within float precision, i.e. that's as close as you can get to zero. It defines the best resolution floats give you.
>Now the overall smallest value is Number.NEGATIVE_INFINITY although that's not really numeric in the strict sense.

## Adding arrays
What if you try to add two arrays?
```js
[1, 2, 3] + [4, 5, 6]  // -> '1,2,34,5,6'
```
>The concatenation happens. Step-by-step it looks like this:
```js
[1, 2, 3] + [4, 5, 6]
// joining
[1, 2, 3].join() + [4, 5, 6].join()
// concatenation
'1,2,3' + '4,5,6'
// ->
'1,2,34,5,6'
```

## `undefined` and `Number`
If we don't pass any argument into the `Number` constructor, we'll get `0`. `undefined` is a value assigned to formal arguments which there are no actual arguments, so you might expect that `Number` without arguments takes `undefined` as a value of its parameter. However, when we pass `undefined`, we will get `NaN`.
```js
Number()          // -> 0
Number(undefined) // -> NaN
```
>According to the specification:
1. If no arguments were passed to this function invocation, let `n` be `+0`.
2. Else, let `n` be ? `ToNumber(value)`.
3. In case with `undefined`, `ToNumber(undefined)` should return `NaN`.

## `parseInt` is a bad guy
```js
parseInt('f*ck');     // -> NaN
parseInt('f*ck', 16); // -> 15
```
>This happens because `parseInt` will continue parsing character-by-character until it hits a character it doesn't know. The `f` in `'f*ck'` is hexadecimal `15`.
```js
parseInt(null, 24) // -> 23
```
> It's converting `null` to the string `"null"` and trying to convert it.

## Math with `true` and `false`
```js
true + true // -> 2
(true + true) * (true + true) - true // -> 3
```
The unary plus operator attempts to convert its value into a number.
```js
Number(true) // -> 1
```

## `[]` and `null` are objects
```js
typeof []   // -> 'object'
typeof null // -> 'object'
// however
null instanceof Object // false
```

## Magically increasing numbers
```js
999999999999999  // -> 999999999999999
9999999999999999 // -> 10000000000000000

10000000000000000       // -> 10000000000000000
10000000000000000 + 1   // -> 10000000000000000
10000000000000000 + 1.1 // -> 10000000000000002
```
>This is caused by IEEE 754-2008 standard for Binary Floating-Point Arithmetic. At this scale, it rounds to the nearest even number.

## Precision of `0.1 + 0.2`
```js
0.1 + 0.2 // -> 0.30000000000000004
(0.1 + 0.2) === 0.3 // -> false
```
> The constants `0.2` and `0.3` in your program will also be approximations to their true values. It happens that the closest `double` to `0.2` is larger than the rational number `0.2` but that the closest `double` to `0.3` is smaller than the rational number `0.3`. The sum of `0.1` and `0.2` winds up being larger than the rational number `0.3` and hence disagreeing with the constant in your code.
>This problem is so known that even there is a website called [0.30000000000000004.com](http://0.30000000000000004.com/). It occurs in every language that uses floating-point math, not just JavaScript.

## Comparison of three numbers
```js
1 < 2 < 3 // -> true
3 > 2 > 1 // -> false
```
The problem is in the first part of an expression. Here's how it works:
```js
1 < 2 < 3 // 1 < 2 -> true
true  < 3 // true -> 1
1     < 3 // -> true

3 > 2 > 1 // 3 > 2 -> true
true  > 1 // true -> 1
1     > 1 // -> false
```


## Funny math

Often the results of an arithmetic operations in JavaScript might be quite unexpectable. Consider these examples:

```js
 3  - 1  // -> 2
 3  + 1  // -> 4
'3' - 1  // -> 2
'3' + 1  // -> '31'

'' + '' // -> ''
[] + [] // -> ''
{} + [] // -> 0
[] + {} // -> '[object Object]'
{} + {} // -> '[object Object][object Object]'

'222' - -'111' // -> 333

[4] * [4]       // -> 16
[] * []         // -> 0
[4, 4] * [4, 4] // NaN
```
What's happening in the first four examples? Here's a small table to understand addition in JavaScript:
```
Number  + Number  -> addition
Boolean + Number  -> addition
Boolean + Boolean -> addition
Number  + String  -> concatenation
String  + Boolean -> concatenation
String  + String  -> concatenation
```
What about the rest examples? A `ToPrimitive` and `ToString` methods are being implicitly called for `[]` and `{}` before addition. 

## Strings aren't instances of `String`
```js
'str' // -> 'str'
typeof 'str' // -> 'string'
'str' instanceof String // -> false

typeof String('str')   // -> 'string'
String('str')          // -> 'str'
String('str') == 'str' // -> true

new String('str') == 'str' // -> true
typeof new String('str')   // -> ‘object'

new String('str') // -> [String: 'str']
```
More information about the String constructor in the specification:
* [**21.1.1** The String Constructor](https://www.ecma-international.org/ecma-262/#sec-string-constructor)

## Insidious `try..catch`
What will this expression return? `2` or `3`?
```js
(() => {
  try {
    return 2;
  } finally {
    return 3;
  }
})()
```
>* [**13.15** The `try` Statement](https://www.ecma-international.org/ecma-262/#sec-try-statement)







