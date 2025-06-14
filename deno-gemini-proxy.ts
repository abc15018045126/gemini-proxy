// deno_index.ts
// 这是一个为 Deno Playground 设计的单一文件 Gemini API 代理，
// 它包含了前端 UI (HTML, CSS, JS) 和后端代理逻辑，
// 并将 Gemini API 响应转换为 OpenAI 兼容格式。

// === 嵌入式前端文件内容 ===

const HTML_CONTENT = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Playground</title>
    <style>${/* Inlining style.css content here */`
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

#app {
    background-color: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    width: 90%;
    max-width: 600px;
}

h1 {
    text-align: center;
    color: #333;
}

textarea {
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-sizing: border-box;
    font-size: 16px;
    resize: vertical;
}

button {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    width: 100%;
    box-sizing: border-box;
}

button:hover {
    background-color: #0056b3;
}

#response {
    margin-top: 20px;
    padding: 10px;
    background-color: #e9e9e9;
    border-radius: 4px;
    min-height: 100px;
    white-space: pre-wrap; /* Preserve whitespace and line breaks */
    word-wrap: break-word; /* Break long words */
}
    `}</style>
</head>
<body>
    <div id="app">
        <h1>Gemini AI Chat</h1>
        <label for="model-select">Select Model:</label>
        <select id="model-select" style="width:100%; padding:8px; margin-bottom:10px;"></select>

        <label for="temperature-input">Temperature (0.0 - 1.0):</label>
        <input type="number" id="temperature-input" min="0" max="1" step="0.1" value="0.7" style="width:100%; padding:8px; margin-bottom:10px;">

        <textarea id="prompt-input" placeholder="Enter your prompt here..." rows="5"></textarea>
        <button id="send-button">Send</button>
        <div id="response"></div>
    </div>
    <script type="module">${/* Inlining index.js content here */`
const promptInput = document.getElementById('prompt-input');
const sendButton = document.getElementById('send-button');
const responseDiv = document.getElementById('response');
const modelSelect = document.getElementById('model-select');
const temperatureInput = document.getElementById('temperature-input');

// Function to fetch models
async function fetchModels() {
    try {
        // Calls the /models endpoint of this Deno Worker
        const response = await fetch('/models');
        if (!response.ok) {
            throw new Error(\`HTTP error! Status: \${response.status}\`);
        }
        const data = await response.json();
        modelSelect.innerHTML = ''; // Clear existing options
        data.data.forEach(model => {
            // Filter for models capable of text generation (adjust based on your needs)
            // The /models endpoint might list more than just generation models.
            // For gemini-playground, it usually picks "generate" models.
            if (model.id.includes('generate') || model.id.includes('gemini-')) {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.id;
                if (model.id === 'gemini-1.5-flash-latest' || model.id === 'gemini-pro') { // Prefer Flash or Pro
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            }
        });
    } catch (error) {
        console.error('Error fetching models:', error);
        alert('Failed to fetch models. Please check console for details.'); // Using alert as per original code
    }
}

// Function to send message to the API
async function sendMessage() {
    const prompt = promptInput.value;
    const model = modelSelect.value;
    const temperature = parseFloat(temperatureInput.value);

    if (!prompt) {
        alert('Please enter a prompt.'); // Using alert as per original code
        return;
    }

    responseDiv.textContent = 'Generating...';
    sendButton.disabled = true;

    try {
        // Calls the /v1/chat/completions endpoint of this Deno Worker
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // API key is handled by the Deno Worker backend, no need here
            },
            body: JSON.stringify({
                model: model,
                messages: [{ role: 'user', content: prompt }],
                temperature: temperature,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(\`HTTP error! Status: \${response.status}, Details: \${errorText}\`);
        }

        const data = await response.json();
        // The Deno Worker transforms Gemini response to OpenAI format,
        // so we expect data.choices[0].message.content
        responseDiv.textContent = data.choices[0].message.content;

    } catch (error) {
        console.error('Error sending message:', error);
        responseDiv.textContent = \`Error: \${error.message}\`;
        alert('An error occurred. Please check console for details.'); // Using alert as per original code
    } finally {
        sendButton.disabled = false;
    }
}

// Event Listeners
sendButton.addEventListener('click', sendMessage);

// Initial model fetch when page loads
document.addEventListener('DOMContentLoaded', fetchModels);
    `}</script>
</body>
</html>
`;

// === 后端代理逻辑 ===

// @deno-types="npm:@types/ws@latest" // For WebSocket types in Deno

// --- 通用工具函数和错误处理 ---

/**
 * Custom Error class for creating HTTP-specific errors.
 * Includes a `status` property for setting HTTP status codes.
 */
class HttpError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

/**
 * Utility function to add necessary CORS (Cross-Origin Resource Sharing) headers
 * to a ResponseInit object or an existing Response object.
 */
const fixCors = (responseInfo: Response | ResponseInit): ResponseInit => {
  const headers = new Headers(responseInfo.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key");

  const status = 'status' in responseInfo ? responseInfo.status : undefined;
  const statusText = 'statusText' in responseInfo ? responseInfo.statusText : undefined;

  return { headers, status, statusText };
};

/**
 * Handles CORS preflight (OPTIONS) requests.
 */
const handleOPTIONS = async (): Promise<Response> => {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key",
      "Access-Control-Max-Age": "86400",
    },
  });
};

// --- Gemini API 相关的常量和辅助函数 ---

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version

// 构建请求头，包含 API 密钥
const makeHeaders = (apiKey: string | null, more?: Record<string, string>): HeadersInit => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

// === OpenAI 兼容性转换逻辑 (从 Gemini 到 OpenAI 格式) ===

const adjustProps = (schemaPart: any) => {
  if (typeof schemaPart !== "object" || schemaPart === null) {
    return;
  }
  if (Array.isArray(schemaPart)) {
    schemaPart.forEach(adjustProps);
  } else {
    if (schemaPart.type === "object" && schemaPart.properties && schemaPart.additionalProperties === false) {
      delete schemaPart.additionalProperties;
    }
    Object.values(schemaPart).forEach(adjustProps);
  }
};

const adjustSchema = (schema: any) => {
  const obj = schema[schema.type];
  delete obj.strict;
  return adjustProps(schema);
};

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));

const fieldsMap = {
  frequency_penalty: "frequencyPenalty",
  max_completion_tokens: "maxOutputTokens",
  max_tokens: "maxOutputTokens",
  n: "candidateCount", // not for streaming
  presence_penalty: "presencePenalty",
  seed: "seed",
  stop: "stopSequences",
  temperature: "temperature",
  top_k: "topK", // non-standard
  top_p: "topP",
};

const transformConfig = (req: any) => {
  let cfg: Record<string, any> = {};
  for (let key in req) {
    const matchedKey = (fieldsMap as Record<string, string>)[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }
  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        adjustSchema(req.response_format);
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  return cfg;
};

// parseImg adjusted for Deno/Browser btoa for base64 encoding
const parseImg = async (url: string): Promise<any> => {
  let mimeType: string, data: string;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type") || 'application/octet-stream';
      const arrayBuffer = await response.arrayBuffer();
      // Convert ArrayBuffer to base64 string
      data = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
    } catch (err: any) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match || !match.groups) {
      throw new HttpError("Invalid image data: " + url, 400);
    }
    ({ mimeType, data } = match.groups);
    if (!mimeType || !data) {
      throw new HttpError("Invalid image data: " + url, 400);
    }
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

const transformFnResponse = ({ content, tool_call_id }: any, parts: any) => {
  if (!parts.calls) {
    throw new HttpError("No function calls found in the previous message", 400);
  }
  let response;
  try {
    response = JSON.parse(content);
  } catch (err) {
    console.error("Error parsing function response content:", err);
    throw new HttpError("Invalid function response: " + content, 400);
  }
  if (typeof response !== "object" || response === null || Array.isArray(response)) {
    response = { result: response };
  }
  if (!tool_call_id) {
    throw new HttpError("tool_call_id not specified", 400);
  }
  const { i, name } = parts.calls[tool_call_id] ?? {};
  if (!name) {
    throw new HttpError("Unknown tool_call_id: " + tool_call_id, 400);
  }
  if (parts[i]) {
    throw new HttpError("Duplicated tool_call_id: " + tool_call_id, 400);
  }
  parts[i] = {
    functionResponse: {
      id: tool_call_id.startsWith("call_") ? null : tool_call_id,
      name,
      response,
    }
  };
};

const transformFnCalls = ({ tool_calls }: any) => {
  const calls: Record<string, { i: number; name: string }> = {};
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type }: any, i: number) => {
    if (type !== "function") {
      throw new HttpError(`Unsupported tool_call type: "${type}"`, 400);
    }
    let args;
    try {
      args = JSON.parse(argstr);
    } catch (err) {
      console.error("Error parsing function arguments:", err);
      throw new HttpError("Invalid function arguments: " + argstr, 400);
    }
    calls[id] = { i, name };
    return {
      functionCall: {
        id: id.startsWith("call_") ? null : id,
        name,
        args,
      }
    };
  });
  (parts as any).calls = calls; // Attach calls for later lookup
  return parts;
};

const transformMsg = async ({ content }: any): Promise<any[]> => {
  const parts: any[] = [];
  if (!Array.isArray(content)) {
    parts.push({ text: content });
    return parts;
  }
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          }
        });
        break;
      default:
        throw new HttpError(`Unknown "content" item type: "${item.type}"`, 400);
    }
  }
  if (content.every((item: any) => item.type === "image_url")) {
    parts.push({ text: "" }); // to avoid "Unable to submit request because it must have a text parameter"
  }
  return parts;
};

const transformMessages = async (messages: any[]): Promise<{ system_instruction?: any; contents: any[] }> => {
  if (!messages) { return { contents: [] }; }
  const contents: any[] = [];
  let system_instruction: any;
  for (const item of messages) {
    switch (item.role) {
      case "system":
        system_instruction = { parts: await transformMsg(item) };
        continue;
      case "tool":
        let { role, parts } = contents[contents.length - 1] ?? {};
        if (role !== "function") {
          const calls = parts?.calls;
          parts = [];
          (parts as any).calls = calls;
          contents.push({
            role: "function", // ignored by Gemini, but for mapping
            parts
          });
        }
        transformFnResponse(item, parts);
        continue;
      case "assistant":
        item.role = "model";
        break;
      case "user":
        break;
      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
    contents.push({
      role: item.role,
      parts: item.tool_calls ? transformFnCalls(item) : await transformMsg(item)
    });
  }
  if (system_instruction) {
    if (!contents[0]?.parts.some((part: any) => part.text)) {
      contents.unshift({ role: "user", parts: [{ text: " " }] }); // Add dummy user message if first content is not text
    }
  }
  return { system_instruction, contents };
};

const transformTools = (req: any) => {
  let tools: any, tool_config: any;
  if (req.tools) {
    const funcs = req.tools.filter((tool: any) => tool.type === "function");
    funcs.forEach(adjustSchema);
    tools = [{ function_declarations: funcs.map((schema: any) => schema.function) }];
  }
  if (req.tool_choice) {
    const allowed_function_names = req.tool_choice?.type === "function" ? [req.tool_choice?.function?.name] : undefined;
    if (allowed_function_names || typeof req.tool_choice === "string") {
      tool_config = {
        function_calling_config: {
          mode: allowed_function_names ? "ANY" : req.tool_choice.toUpperCase(),
          allowed_function_names
        }
      };
    }
  }
  return { tools, tool_config };
};

const transformRequest = async (req: any): Promise<any> => ({
  ...await transformMessages(req.messages),
  safetySettings,
  generationConfig: transformConfig(req),
  ...transformTools(req),
});

const generateId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap: Record<string, string> = {
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
};
const SEP = "\n\n|>";

const transformCandidates = (key: string, cand: any) => {
  const message: any = { role: "assistant", content: [] };
  for (const part of cand.content?.parts ?? []) {
    if (part.functionCall) {
      const fc = part.functionCall;
      message.tool_calls = message.tool_calls ?? [];
      message.tool_calls.push({
        id: fc.id ?? "call_" + generateId(),
        type: "function",
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.args),
        }
      });
    } else if (part.text) { // Ensure text part exists
      message.content.push(part.text);
    }
  }
  message.content = message.content.join(SEP) || null;
  return {
    index: cand.index || 0,
    [key]: message,
    logprobs: null,
    finish_reason: message.tool_calls ? "tool_calls" : reasonsMap[cand.finishReason] || cand.finishReason,
  };
};
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data: any) => ({
  completion_tokens: data.candidatesTokenCount,
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount
});

const checkPromptBlock = (choices: any[], promptFeedback: any, key: string) => {
  if (choices.length) { return; }
  if (promptFeedback?.blockReason) {
    console.log("Prompt block reason:", promptFeedback.blockReason);
    if (promptFeedback.blockReason === "SAFETY") {
      promptFeedback.safetyRatings
        .filter((r: any) => r.blocked)
        .forEach((r: any) => console.log(r));
    }
    choices.push({
      index: 0,
      [key]: null,
      finish_reason: "content_filter",
    });
  }
  return true;
};

const processCompletionsResponse = (data: any, model: string, id: string) => {
  const obj: any = {
    id,
    choices: data.candidates.map(transformCandidatesMessage),
    created: Math.floor(Date.now() / 1000),
    model: data.modelVersion ?? model,
    object: "chat.completion",
    usage: data.usageMetadata && transformUsage(data.usageMetadata),
  };
  if (obj.choices.length === 0) {
    checkPromptBlock(obj.choices, data.promptFeedback, "message");
  }
  return JSON.stringify(obj);
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
function parseStream(chunk: string, controller: TransformStreamDefaultController<string>) {
  // @ts-ignore
  this.buffer = (this.buffer || "") + chunk;
  // @ts-ignore
  do {
    // @ts-ignore
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    controller.enqueue(match[1]);
    // @ts-ignore
    this.buffer = this.buffer.substring(match[0].length);
  } while (true);
}
function parseStreamFlush(controller: TransformStreamDefaultController<string>) {
  // @ts-ignore
  if (this.buffer) {
    // @ts-ignore
    console.error("Invalid data:", this.buffer);
    // @ts-ignore
    controller.enqueue(this.buffer);
    // @ts-ignore
    this.shared.is_buffers_rest = true;
  }
}

const delimiter = "\n\n";
const sseline = (obj: any) => {
  obj.created = Math.floor(Date.now() / 1000);
  return "data: " + JSON.stringify(obj) + delimiter;
};
function toOpenAiStream(line: string, controller: TransformStreamDefaultController<string>) {
  let data: any;
  try {
    data = JSON.parse(line);
    if (!data.candidates) {
      throw new Error("Invalid completion chunk object");
    }
  } catch (err) {
    console.error("Error parsing response:", err);
    // @ts-ignore
    if (!this.shared.is_buffers_rest) { line =+ delimiter; }
    controller.enqueue(line);
    return;
  }
  const obj: any = {
    // @ts-ignore
    id: this.id,
    choices: data.candidates.map(transformCandidatesDelta),
    // @ts-ignore
    model: data.modelVersion ?? this.model,
    object: "chat.completion.chunk",
    // @ts-ignore
    usage: data.usageMetadata && this.streamIncludeUsage ? null : undefined,
  };
  // @ts-ignore
  if (checkPromptBlock(obj.choices, data.promptFeedback, "delta")) {
    controller.enqueue(sseline(obj));
    return;
  }
  console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
  const cand = obj.choices[0];
  cand.index = cand.index || 0;
  const finish_reason = cand.finish_reason;
  cand.finish_reason = null;
  // @ts-ignore
  if (!this.last[cand.index]) {
    controller.enqueue(sseline({
      ...obj,
      choices: [{ ...cand, tool_calls: undefined, delta: { role: "assistant", content: "" } }],
    }));
  }
  delete cand.delta.role;
  if ("content" in cand.delta) {
    controller.enqueue(sseline(obj));
  }
  cand.finish_reason = finish_reason;
  // @ts-ignore
  if (data.usageMetadata && this.streamIncludeUsage) {
    obj.usage = transformUsage(data.usageMetadata);
  }
  cand.delta = {};
  // @ts-ignore
  this.last[cand.index] = obj;
}
function toOpenAiStreamFlush(controller: TransformStreamDefaultController<string>) {
  // @ts-ignore
  if (this.last.length > 0) {
    // @ts-ignore
    for (const obj of this.last) {
      controller.enqueue(sseline(obj));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}

// === API 处理函数 (使用上述转换逻辑) ===

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings(reqBody: any, apiKey: string | null): Promise<Response> {
  if (typeof reqBody.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  let model: string;
  if (reqBody.model.startsWith("models/")) {
    model = reqBody.model;
  } else {
    if (!reqBody.model.startsWith("gemini-")) {
      reqBody.model = DEFAULT_EMBEDDINGS_MODEL;
    }
    model = "models/" + reqBody.model;
  }
  if (!Array.isArray(reqBody.input)) {
    reqBody.input = [reqBody.input];
  }
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": reqBody.input.map((text: string) => ({
        model,
        content: { parts: [{ text }] },
        outputDimensionality: reqBody.dimensions,
      }))
    })
  });
  let body: any = response.body;
  if (response.ok) {
    body = JSON.parse(await response.text());
    const { embeddings } = body;
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }: { values: number[] }, index: number) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: reqBody.model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-1.5-flash-latest"; // Updated default model for broader compatibility
async function handleCompletions(reqBody: any, apiKey: string | null): Promise<Response> {
  let model = DEFAULT_MODEL;
  switch (true) {
    case typeof reqBody.model !== "string":
      break;
    case reqBody.model.startsWith("models/"):
      model = reqBody.model.substring(7);
      break;
    case reqBody.model.startsWith("gemini-"):
    case reqBody.model.startsWith("gemma-"):
    case reqBody.model.startsWith("learnlm-"):
      model = reqBody.model;
      break;
  }
  let body = await transformRequest(reqBody);
  switch (true) {
    case model.endsWith(":search"):
      model = model.substring(0, model.length - 7);
      body.tools = body.tools || [];
      body.tools.push({ googleSearch: {} });
      break;
    case reqBody.model.endsWith("-search-preview"): // Legacy search model
      body.tools = body.tools || [];
      body.tools.push({ googleSearch: {} });
      break;
  }
  const TASK = reqBody.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (reqBody.stream) { url += "?alt=sse"; }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  body = response.body;
  if (response.ok) {
    let id = "chatcmpl-" + generateId();
    const shared = {}; // Shared state for transform streams

    if (reqBody.stream) {
      body = response.body!
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          // @ts-ignore
          buffer: "",
          shared,
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          // @ts-ignore
          streamIncludeUsage: reqBody.stream_options?.include_usage,
          model, id, last: [],
          shared,
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        body = JSON.parse(body);
        if (!body.candidates) {
          throw new Error("Invalid completion object");
        }
      } catch (err) {
        console.error("Error parsing response:", err);
        return new Response(body, fixCors(response)); // output as is
      }
      body = processCompletionsResponse(body, model, id);
    }
  }
  return new Response(body, fixCors(response));
}

async function handleModels(apiKey: string | null): Promise<Response> {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let body: any = response.body;
  if (response.ok) {
    body = JSON.parse(await response.text());
    const { models } = body;
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }: { name: string }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "google", // Changed for clarity
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

// --- Deno WebSocket 代理处理 ---

/**
 * Handles WebSocket upgrade requests and proxies them to the target Gemini WebSocket API.
 * @param {Request} req The incoming WebSocket upgrade request.
 * @returns {Promise<Response>} A Promise that resolves to the WebSocket upgrade response.
 */
async function handleWebSocket(req: Request): Promise<Response> {
  const { socket: clientWs, response } = Deno.upgradeWebSocket(req);

  const url = new URL(req.url);
  const targetUrl = `wss://generativelanguage.googleapis.com${url.pathname}${url.search}`;

  console.log('Target WebSocket URL:', targetUrl);

  const pendingMessages: string[] = [];
  let targetWs: WebSocket;

  try {
    targetWs = new WebSocket(targetUrl);
  } catch (error) {
    console.error('Failed to create target WebSocket:', error);
    return new Response('Failed to connect to target WebSocket.', { status: 500 });
  }

  targetWs.onopen = () => {
    console.log('Connected to Gemini WebSocket API');
    pendingMessages.forEach(msg => targetWs.send(msg));
    pendingMessages.length = 0;
  };

  clientWs.onmessage = (event) => {
    console.log('Client WebSocket message received');
    if (targetWs.readyState === WebSocket.OPEN) {
      targetWs.send(event.data);
    } else {
      pendingMessages.push(event.data.toString());
    }
  };

  targetWs.onmessage = (event) => {
    console.log('Gemini WebSocket message received');
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(event.data);
    }
  };

  clientWs.onclose = (event) => {
    console.log('Client WebSocket connection closed');
    if (targetWs.readyState === WebSocket.OPEN) {
      targetWs.close(event.code, event.reason);
    }
  };

  targetWs.onclose = (event) => {
    console.log('Gemini WebSocket connection closed');
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.close(event.code, event.reason);
    }
  };

  targetWs.onerror = (error) => {
    console.error('Gemini WebSocket error:', error);
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ error: 'Gemini WebSocket error occurred.' }));
      clientWs.close(1011, 'Internal Server Error');
    }
  };

  return response;
}

// === 主请求处理器 ===

/**
 * Main request handler for the Deno server.
 * Routes incoming requests based on their type (WebSocket, API, Static File).
 */
async function handleRequest(req: Request): Promise<Response> {
  const url = new URL(req.url);
  console.log('Incoming Request URL:', req.url);

  // 获取 API Key (假设从 Authorization 头或 URL 参数中获取)
  // 为了 Deno Playground 简单运行，这里直接从环境变量读取，
  // 或者您可以在 Deno Playground 的 "Environment Variables" 设置 `GEMINI_API_KEY`
  const apiKey = Deno.env.get("GEMINI_API_KEY") || req.headers.get("Authorization")?.split(" ")[1] || url.searchParams.get("key");

  if (!apiKey) {
    return new Response(JSON.stringify({ error: "API Key is missing. Please provide it via Authorization header or 'key' URL parameter, or set GEMINI_API_KEY environment variable." }), {
      status: 401,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      }
    });
  }

  // 1. WebSocket 处理：检查 Upgrade 头
  if (req.headers.get("Upgrade")?.toLowerCase() === "websocket") {
    return handleWebSocket(req);
  }

  // 2. API 请求处理：检查路径是否匹配 Gemini API 路径
  // Updated to include /v1/chat/completions
  if (url.pathname.endsWith("/v1/chat/completions") || url.pathname.endsWith("/chat/completions")) {
    const reqBody = await req.json();
    return handleCompletions(reqBody, apiKey);
  } else if (url.pathname.endsWith("/embeddings")) {
    const reqBody = await req.json();
    return handleEmbeddings(reqBody, apiKey);
  } else if (url.pathname.endsWith("/models")) {
    return handleModels(apiKey);
  }

  // 3. 静态文件处理：提供嵌入的 HTML 内容
  // 仅仅针对根路径 `/` 或 `/index.html` 返回 HTML 内容
  if (url.pathname === '/' || url.pathname === '/index.html') {
    return new Response(HTML_CONTENT, {
      headers: {
        'content-type': 'text/html;charset=UTF-8',
      },
    });
  }

  // 如果以上都不匹配，返回 404
  return new Response('Not Found', {
    status: 404,
    headers: {
      'content-type': 'text/plain;charset=UTF-8',
    }
  });
}

// 启动 Deno 服务器
Deno.serve(handleRequest);
