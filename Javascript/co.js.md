Generator based control flow goodness for nodejs and the browser, using promises, letting you write non-blocking code in a nice-ish way.

generator function executor:
```js
function co(gen){
  const g = gen();

  function next(data){
    const { value, done } = g.next(data);
    if (done) return;
    next(value);
  }

  next();
}
```

generator function executor of Promise chain:
```js
function co(gen){
  const g = gen();

  function next(data){
    const { value, done } = g.next(data);
    if (done) return value;
    value.then(data => {
      next(data);
    });
  }

  next();
}
```