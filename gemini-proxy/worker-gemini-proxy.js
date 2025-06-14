// Cloudflare Worker - Gemini API 反向代理

// 定义目标 Gemini API 的基础 URL。
const GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com";

// Worker 的主要事件监听器。当 Worker 收到一个 fetch 事件时触发。
addEventListener("fetch", event => {
  // 阻止 Worker 的默认行为，并用我们自定义的响应来代替。
  event.respondWith(handleRequest(event.request));
});

/**
 * 处理传入的请求。
 * @param {Request} request 传入的 HTTP 请求对象。
 * @returns {Promise<Response>} Promise 解析为一个 Response 对象。
 */
async function handleRequest(request) {
  // 从传入请求的 URL 创建一个 URL 对象。
  const url = new URL(request.url);

  // 构建目标 Gemini API 的完整 URL。
  // 它通过拼接 Gemini API 的基础 URL 和传入请求的路径名来完成。
  // 传入请求的查询参数（包括 API 密钥）会自动包含在 url.search 中。
  const targetUrl = new URL(url.pathname + url.search, GEMINI_API_BASE_URL);

  console.log(`代理请求到: ${targetUrl.toString()}`);

  try {
    // 使用 fetch API 将传入的请求转发到目标 Gemini API。
    // request.method：保留原始请求方法（GET, POST, PUT, DELETE 等）。
    // request.headers：保留所有原始请求头，这对于内容类型和授权头很重要。
    // request.body：保留原始请求体，这对于 POST 请求的数据很重要。
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });

    console.log(`收到 Gemini API 响应，状态码: ${response.status}`);

    // 返回从 Gemini API 收到的响应给客户端。
    // response.headers：保留所有原始响应头。
    // response.body：保留原始响应体。
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } catch (error) {
    // 如果在转发请求或接收响应时发生错误，则捕获并记录错误。
    console.error("代理请求失败:", error);
    // 返回一个带有错误信息的 500 内部服务器错误响应。
    return new Response(`代理请求失败: ${error.message}`, { status: 500 });
  }
}
