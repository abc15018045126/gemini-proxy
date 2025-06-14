// main.ts
// 这是一个 Deno 应用程序，用于作为 Gemini API 的反向代理。

// 从 Deno 标准库导入 serve 函数，用于创建 HTTP 服务器。
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

// 定义目标 Gemini API 的基础 URL。
const GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com";

// 处理所有传入的 HTTP 请求的异步函数。
async function handler(req: Request): Promise<Response> {
  // 从传入请求的 URL 创建一个 URL 对象。
  const url = new URL(req.url);

  // 构建目标 Gemini API 的完整 URL。
  // 它通过拼接 Gemini API 的基础 URL 和传入请求的路径名来完成。
  // 传入请求的查询参数（包括 API 密钥）会自动包含在 req.url 中。
  const targetUrl = new URL(url.pathname + url.search, GEMINI_API_BASE_URL);

  console.log(`代理请求到: ${targetUrl.toString()}`);

  try {
    // 使用 fetch API 将传入的请求转发到目标 Gemini API。
    // req.method：保留原始请求方法（GET, POST, PUT, DELETE 等）。
    // req.headers：保留所有原始请求头，这对于内容类型和授权头很重要。
    // req.body：保留原始请求体，这对于 POST 请求的数据很重要。
    const response = await fetch(targetUrl, {
      method: req.method,
      headers: req.headers,
      body: req.body, // 如果是 GET 请求，body 将为 null，fetch 会自动处理。
      // duplex: 'half' 是为了处理请求体流，确保正确转发。
      duplex: 'half' as any, // Deno 的 fetch 类型定义可能需要此类型断言
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

// 启动 Deno HTTP 服务器。Deno Deploy 会自动管理端口。
console.log(`Deno Gemini API 代理服务器正在运行。`);
serve(handler);
