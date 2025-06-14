import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
function denoGeminiProxyMin() {
  // ...existing code...
  

  const GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com";

  async function handler(req: Request): Promise<Response> {
    // ...existing code...
    const url = new URL(req.url);
    const targetUrl = new URL(url.pathname + url.search, GEMINI_API_BASE_URL);

    console.log(`代理请求到: ${targetUrl.toString()}`);

    try {
      const response = await fetch(targetUrl, {
        method: req.method,
        headers: req.headers,
        body: req.body,
        duplex: 'half' as any,
      });

      console.log(`收到 Gemini API 响应，状态码: ${response.status}`);

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } catch (error) {
      console.error("代理请求失败:", error);
      return new Response(`代理请求失败: ${error.message}`, { status: 500 });
    }
  }
  return handler;

  


}

const handler = denoGeminiProxyMin(); 
console.log(`Deno Gemini API 代理服务器正在运行。`);
serve(handler);


