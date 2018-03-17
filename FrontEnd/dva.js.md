> https://github.com/dvajs/dva/blob/master/docs/Concepts_zh-CN.md
> https://github.com/dvajs/dva/blob/master/docs/API_zh-CN.md


# Reducer
reducer 是一个函数，接受 state 和 action，返回老的或新的 state 。即：(state, action) => state


# Effect

Effects

put

用于触发 action 。

yield put({ type: 'todos/add', payload: 'Learn Dva' });
call

用于调用异步逻辑，支持 promise 。

const result = yield call(fetch, '/todos');
select

用于从 state 里获取数据。

const todos = yield select(state => state.todos);


错误处理
dva 里，effects 和 subscriptions 的抛错全部会走 onError hook，所以可以在 onError 里统一处理错误。

const app = dva({
  onError(e, dispatch) {
    console.log(e.message);
  },
});

然后 effects 里的抛错和 reject 的 promise 就都会被捕获到了


# Subscription
subscriptions 是订阅，用于订阅一个数据源，然后根据需要 dispatch 相应的 action。数据源可以是当前的时间、服务器的 websocket 连接、keyboard 输入、geolocation 变化、history 路由变化等等。格式为 ({ dispatch, history }) => unsubscribe 。





功能调整基本都可以按照以下三步进行：

1. service
2. model
3. component


