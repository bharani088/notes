The JavaScript runtime you’ve got is ether JavaScriptCore (non-debug) or V8 (debug). Even though you can use NPM and a node server is running on the background, your code does not actually run on Node.JS. So you won’t be able to use of the Node.JS packages. A typical example is jsonwebtoken, which uses NodeJS’s crypto module.

just like it's not possible to use node modules in Chrome or Firefox

For emphasis: just as Node and the browser are different environments that happen to use the same language, React Native is a third JS environment that's not a browser and isn't Node. Packages need to be written to either be universal or specifically written for React Native.

There's the "react-native" key in package.json for aliasing modules but fundamentally packages made for Node aren't necessarily going to work with RN.


http://ruoyusun.com/2015/11/01/things-i-wish-i-were-told-about-react-native.html

https://github.com/facebook/react-native/issues/5881


React-Native的Fetch在Android上的底层实现是依赖OKHttp的。

http://www.jianshu.com/p/8089edd88655

XMLHttpRequest API 是在 iOS networking apis 之上实现的。