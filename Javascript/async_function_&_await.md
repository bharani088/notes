# async function

When an async function is called, it returns a Promise. When the async function returns a value, the Promise will be resolved with the returned value.  When the async function throws an exception or some value, the Promise will be rejected with the thrown value.

An async function can contain an await expression, that pauses the execution of the async function and waits for the passed Promise's resolution, and then resumes the async function's execution and returns the resolved value.

## Rewriting a promise chain with an async function

An API that returns a Promise will result in a promise chain, and it splits the function into many parts. Consider the following code:
```js
function getProcessedData(url) {
  return downloadData(url) // returns a promise
    .catch(e => {
      return downloadFallbackData(url); // returns a promise
    })
    .then(v => {
      return processDataInWorker(v); // returns a promise
    });
}
```
it can be rewritten with a single async function as follows:
```js
async function getProcessedData(url) {
  let v;
  try {
    v = await downloadData(url); 
  } catch(e) {
    v = await downloadFallbackData(url);
  }
  return processDataInWorker(v);
}
```
Note that in the above example, there is no await statement on the return statement, because the return value of an async function is implicitly wrapped in Promise.resolve.


# await

The await operator is used to wait for a Promise. It can only be used inside an async function.
```js
[rv] = await expression;
```

**expression**
A Promise or any value to wait for the resolution.

**rv**
Returns the resolved value of the promise, or the value itself if it's not a Promise.

The await expression causes async function execution to pause, to wait for the Promise's resolution, and to resume the async function execution when the value is resolved. It then returns the resolved value. If the value is not a Promise, it's converted to a resolved Promise.

If the Promise is rejected, the await expression throws the rejected value.