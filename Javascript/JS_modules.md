The problem with CommonJs style module definition is that it is synchronous.
A few side notes -- Why can't we just copy an existing module system from another language? Because JS needs **asynchronous** module loading!!

FYI, CommonJS in Node is synchronous, so require() is slow.
```js
;(function(name, definition) {
  'use strict';

  var hasDefine = (typeof define === 'function'),
      hasExports = (typeof module !== 'undefined' && module.exports);

  if (hasDefine) {
    define(name, definition);
  } else if (hasExports) { // CommonJS
    module.exports = definition(require);
  } else {
    this[name] = definition;  // mount to root, e.g. window.mymodule
  }
  
})('module-name', function(require) {
  var module = function() {
    // module code goes here
  }
  return module;
});
```

## CommonJS spec

The project was started by Mozilla engineer Kevin Dangoor in January 2009 and initially named ServerJS.[1]

In August 2009, the project was renamed CommonJS to show the broader applicability of the APIs.[2] Specifications are created and approved in an open process. A specification is only considered final after it has been finished by multiple implementations.[3] CommonJS is not affiliated with the Ecma International group TC39 working on ECMAScript, but some members of TC39 participate in the project.[4]

In May 2013, Isaac Z. Schlueter, the author of npm, the package manager for Node.js, said CommonJS is being made obsolete by Node.js, and is avoided by the core Node.js developers.[5]

## AMD spec & RequireJS

Asynchronous module definition (AMD) is a specification for the programming language JavaScript. It defines an application programming interface (API) that defines code modules and their dependencies, and loads them asynchronously if desired.

The AMD specification is implemented by Dojo Toolkit, RequireJS, and ScriptManJS.

RequireJS is a JavaScript file and module loader. It is optimized for in-browser use, but it can be used in other JavaScript environments, like Rhino and Node. Using a modular script loader like RequireJS will improve the speed and quality of your code.

AMD syntax is too verbose. Since everything is wrapped in ‘define’ function, there are extra indentation for our code. With a small files, it is not much problem, but for a large code base, it can be mentally taxing.
With current browsers(HTTP 1.1), loading many small files can degrade the performance.

## Browserify

Browserify is an open-source JavaScript tool that allows developers to write Node.js-style modules that compile for use in the browser.[1]

Unlike RequireJS, Browserify is a command line tool.

## ES2015 Modules

The static nature of the import and export directive allows static analyzers to build a full tree of dependencies without running code. ES2015 does not support dynamic loading of modules, but a draft specification does.

This solution, by virtue of being integrated in the language, lets runtimes pick the best loading strategy for modules. In other words, when asynchronous loading gives benefits, it can be used by the runtime.

## Webpack

Webpack is an open-source JavaScript module bundler. Webpack takes modules with dependencies and generates static assets representing those modules.[1] It takes the dependencies and generates a dependency graph allowing web developers to use a modular approach for their web application development purposes. The bundler can be used from the command line, or can be configured using a config file which is named webpack.config.js.[2]

Node.js is required for installing webpack. Another important aspect about webpack is that it is highly extensible by the use of loaders. Loaders allow developers to write custom tasks that they want to perform when bundling files together.

Webpack provides code on demand using the moniker code splitting. The Technical Committee 39 for ECMAScript is working on standardization of a function that loads additional code: proposal-dynamic-import.

