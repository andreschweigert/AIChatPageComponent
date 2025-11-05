<?php

// Start output buffering to prevent any stray output from corrupting JSON response
ob_start();

use ILIAS\Plugin\pcaic\Model\ChatConfig;
use ILIAS\Plugin\pcaic\Model\ChatSession;
use ILIAS\Plugin\pcaic\Model\ChatMessage;
use ILIAS\Plugin\pcaic\Model\Attachment;
use ILIAS\Plugin\pcaic\Validation\FileUploadValidator;

/**
 * AI Chat Page Component REST API
 * 
 * RESTful API endpoint for AI chat interactions in ILIAS pages.
 * Provides clean, session-based architecture with automatic configuration management.
 * 
 * ## Architecture
 * 
 * **Clean API Design:**
 * - Frontend sends only `chat_id` + `message` + optional `attachment_ids`
 * - Backend loads all configuration from ChatConfig model
 * - User sessions managed automatically via ChatSession
 * - Messages linked to sessions for proper conversation flow
 * - File attachments handled via ILIAS ResourceStorage
 * 
 * ## Supported Actions
 * 
 * **POST /api.php**
 * - `send_message`: Send message to AI with optional file attachments
 * - `get_messages`: Retrieve conversation history for current user session
 * - `upload_file`: Upload file attachment for multimodal AI analysis
 * 
 * **GET /api.php** 
 * - `get_messages`: Retrieve message history (alternative to POST)
 * - `get_chat_config`: Get chat configuration details
 * 
 * ## Request Format
 * 
 * ```json
 * {
 *   "action": "send_message",
 *   "chat_id": "unique_chat_identifier",
 *   "message": "User message text",
 *   "attachment_ids": [123, 456] // Optional file attachments
 * }
 * ```
 * 
 * ## Response Format
 * 
 * ```json
 * {
 *   "success": true,
 *   "message": "AI response text",
 *   "messages": [...], // Full conversation history
 *   "attachments": [...] // File attachment metadata
 * }
 * ```
 * 
 * ## Security
 * - ILIAS authentication required
 * - User session validation
 * - Input sanitization and validation
 * - File upload security checks
 * - Component-specific logging
 * 
 * ## Error Handling
 * - HTTP status codes (401, 400, 500)
 * - Structured error responses
 * - Comprehensive logging for debugging
 *
 * @author Nadimo Staszak <nadimo.staszak@uni-koeln.de>
 * 
 * @see     ChatConfig For configuration management
 * @see     ChatSession For user session handling
 * @see     ChatMessage For conversation persistence
 */

// Initialize ILIAS environment for standalone API endpoint
$ilias_root = rtrim(dirname(__DIR__, 7), '/');
chdir($ilias_root);
require_once($ilias_root . '/Services/Init/classes/class.ilInitialisation.php');
ilContext::init(ilContext::CONTEXT_WEB);
ilInitialisation::initILIAS();

// Load plugin dependencies and AI service integrations
require_once(__DIR__ . '/src/bootstrap.php');
require_once(__DIR__ . '/classes/ai/class.AIChatPageComponentLLM.php');
require_once(__DIR__ . '/classes/ai/class.AIChatPageComponentRAMSES.php');
require_once(__DIR__ . '/classes/ai/class.AIChatPageComponentOpenAI.php');

global $DIC;
$logger = $DIC->logger()->comp('pcaic');

// Configure HTTP response for JSON API
header('Content-Type: application/json');
header('Cache-Control: no-cache, no-store, must-revalidate');

try {
    // Enforce ILIAS authentication - reject anonymous access
    if (!$DIC->user() || $DIC->user()->getId() == ANONYMOUS_USER_ID) {
        sendApiResponse(['error' => 'Authentication required'], 401);
    }

    $method = $_SERVER['REQUEST_METHOD'];
    $request = $DIC->http()->request();
    
    if ($method === 'GET') {
        $queryParams = $request->getQueryParams();
        $data = is_array($queryParams) ? $queryParams : $queryParams->toArray();
    } elseif ($method === 'POST') {
        // Check if this is a file upload (multipart/form-data)
        $content_type = $_SERVER['CONTENT_TYPE'] ?? '';
        if (strpos($content_type, 'multipart/form-data') !== false) {
            // File upload - use $_POST data, convert ILIAS wrapper to array
            $postData = $request->getParsedBody();
            $data = is_array($postData) ? $postData : $postData->toArray();
        } else {
            // Regular JSON request
            $input = file_get_contents('php://input');
            $data = json_decode($input, true);
            if ($data === null) {
                // Fallback to form data
                $postData = $request->getParsedBody();
                $data = is_array($postData) ? $postData : $postData->toArray();
            }
        }
    } else {
        sendApiResponse(['error' => 'Method not allowed'], 405);
    }

    $action = $data['action'] ?? '';

    switch ($action) {
        case 'send_message':
            $result = handleSendMessage($data);
            break;
        case 'send_message_stream':
            handleSendMessageStream($data);
            return; // Stream response, no JSON output
        case 'upload_file':
            $result = handleFileUpload($data);
            break;
        case 'get_upload_config':
            $result = getUploadConfig($data);
            break;
        case 'get_global_config':
            $result = getGlobalConfig($data);
            break;
        case 'load_chat':
            $result = handleLoadChat($data);
            break;
        case 'clear_chat':
            $result = handleClearChat($data);
            break;
        case 'test':
            $result = ['test' => 'API is working', 'timestamp' => time()];
            break;
        default:
            $result = ['error' => 'Invalid action'];
    }
    
    sendApiResponse($result);

} catch (\Exception $e) {
    $logger->error("API request failed", ['error' => $e->getMessage(), 'action' => $data['action'] ?? 'unknown']);
    sendApiResponse(['error' => 'Internal server error'], 500);
}
/**
 * Handle load chat - Load user's session for a chat
 */
function handleLoadChat(array $data): array
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    $chat_id = $data['chat_id'] ?? '';
    if (empty($chat_id)) {
        return ['error' => 'Missing chat_id parameter'];
    }
    
    $user_id = $DIC->user()->getId();
    
    try {
        // Load chat configuration
        $chatConfig = new ChatConfig($chat_id);
        if (!$chatConfig->exists()) {
            return ['error' => 'Chat configuration not found'];
        }
        
        // Find user's session for this chat
        $session = ChatSession::findForUserAndChat($user_id, $chat_id);
        if (!$session) {
            // Create new session if none exists
            $session = ChatSession::createForUserAndChat($user_id, $chat_id);
            $session->save();
        }
        
        // Load messages
        $messages = $session->getMessages();
        
        return [
            'success' => true,
            'config' => $chatConfig->toArray(),
            'session' => $session->toArray(),
            'messages' => array_map(fn($msg) => $msg->toArray(), $messages)
        ];
        
    } catch (\Exception $e) {
        $logger->warning("Failed to load chat configuration", ['chat_id' => $chat_id, 'error' => $e->getMessage()]);
        return ['error' => 'Failed to load chat'];
    }
}

/**
 * Handle clear chat - Clear user's session for a chat
 */
function handleClearChat(array $data): array
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    $chat_id = $data['chat_id'] ?? '';
    if (empty($chat_id)) {
        return ['error' => 'Missing chat_id parameter'];
    }
    
    $user_id = $DIC->user()->getId();
    
    try {
        // Find user's session for this chat
        $session = ChatSession::findForUserAndChat($user_id, $chat_id);
        if ($session) {
            // Delete session and all its messages
            $session->delete();
        }
        
        return [
            'success' => true,
            'message' => 'Chat cleared successfully'
        ];
        
    } catch (\Exception $e) {
        $logger->warning("Failed to clear chat", ['chat_id' => $chat_id, 'error' => $e->getMessage()]);
        return ['error' => 'Failed to clear chat'];
    }
}

/**
 * Handle send message with separated context structure
 */
function handleSendMessage(array $data): array
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    // Extract required data from frontend
    $chat_id = $data['chat_id'] ?? '';
    $user_message = $data['message'] ?? '';
    $attachment_ids = $data['attachment_ids'] ?? [];
    
    if (empty($chat_id) || empty($user_message)) {
        return ['error' => 'Missing required parameters: chat_id and message'];
    }
    
    // Validate message length against configuration
    $char_limit = (int)(\platform\AIChatPageComponentConfig::get('characters_limit') ?: 2000);
    if (strlen($user_message) > $char_limit) {
        return ['error' => sprintf('Message too long. Maximum %d characters allowed.', $char_limit)];
    }
    
    // Ensure attachment_ids is an array
    if (!is_array($attachment_ids)) {
        $attachment_ids = [];
    }
    
    $user_id = $DIC->user()->getId();
    
    try {
        // Load chat configuration (PageComponent settings)
        $chatConfig = new ChatConfig($chat_id);
        if (!$chatConfig->exists()) {
            return ['error' => 'Chat configuration not found'];
        }
        
        // Get or create user session for this chat
        $session = ChatSession::getOrCreateForUserAndChat($user_id, $chat_id);
        
        // Add user message to session
        $userMessage = $session->addMessage('user', $user_message);
        
        // Bind attachments to the message if any
        if (!empty($attachment_ids)) {
            foreach ($attachment_ids as $attachment_id) {
                if (!empty($attachment_id)) {
                    try {
                        $db = $DIC->database();
                        $db->update('pcaic_attachments',
                            ['message_id' => ['integer', $userMessage->getMessageId()]],
                            ['id' => ['integer', $attachment_id]]
                        );
                    } catch (\Exception $e) {
                        $logger->warning("Failed to bind attachment to message", ['attachment_id' => $attachment_id, 'message_id' => $userMessage->getMessageId(), 'error' => $e->getMessage()]);
                    }
                }
            }
        }
        
        // Get recent messages for AI context (respecting max_memory)
        $recentMessages = ChatMessage::getRecentForSession(
            $session->getSessionId(), 
            $chatConfig->getMaxMemory()
        );
        
        // Build CLEAN system prompt (ONLY AI behavior, no content)
        $clean_system_prompt = $chatConfig->getSystemPrompt();

        
        // Build context resources (ALL context as structured resources)
        $contextResources = [];
        
        // Add page context as resource if enabled
        if ($chatConfig->isIncludePageContext()) {
            
            $page_context = getPageContextForChat($chatConfig);
            if (!empty($page_context)) {
                $contextResources[] = [
                    'kind' => 'page_context',
                    'id' => 'current-page',
                    'title' => 'Aktuelle Lernseite',
                    'content' => $page_context
                ];
            }
        }
        
        // Add background files as resources 
        $background_files = $chatConfig->getBackgroundFiles();
        if (!empty($background_files)) {
            $irss = $DIC->resourceStorage();
            
            foreach ($background_files as $file_id) {
                try {
                    $identification = $irss->manage()->find($file_id);
                    if ($identification === null) continue;
                    
                    $revision = $irss->manage()->getCurrentRevision($identification);
                    if ($revision === null) continue;
                    
                    $suffix = strtolower($revision->getInformation()->getSuffix());
                    $mime_type = $revision->getInformation()->getMimeType();
                    
                    // Process ALL file types as structured resources
                    if (in_array($suffix, ['txt', 'md', 'csv'])) {
                        // Text files
                        $content = extractFileContentFromIRSS($identification);
                        if (!empty($content)) {
                            $contextResources[] = [
                                'kind' => 'text_file',
                                'id' => 'bg-text-' . $file_id,
                                'title' => $revision->getTitle(),
                                'mime_type' => $mime_type,
                                'content' => $content
                            ];
                        }
                    } elseif (in_array($suffix, ['jpg', 'jpeg', 'png', 'gif', 'webp'])) {
                        // Single images
                        $content = extractFileContentFromIRSS($identification);
                        if (!empty($content)) {
                            $contextResources[] = [
                                'kind' => 'image_file',
                                'id' => 'bg-img-' . $file_id,
                                'title' => $revision->getTitle(),
                                'mime_type' => $mime_type,
                                'url' => $content
                            ];
                        }
                    } elseif ($suffix === 'pdf') {
                        // PDF pages (converted to images)
                        $content = extractFileContentFromIRSS($identification);
                        if (!empty($content) && is_array($content)) {
                            foreach ($content as $pageIndex => $pageDataUrl) {
                                $contextResources[] = [
                                    'kind' => 'pdf_page',
                                    'id' => 'bg-pdf-' . $file_id . '-page-' . ($pageIndex + 1),
                                    'title' => $revision->getTitle() . ' (Seite ' . ($pageIndex + 1) . ')',
                                    'mime_type' => 'image/png', // converted to PNG
                                    'page_number' => $pageIndex + 1,
                                    'source_file' => $revision->getTitle(),
                                    'url' => $pageDataUrl
                                ];
                            }
                        }
                    }
                } catch (\Exception $e) {
                    $logger->warning("Background file processing failed", ['file_info' => $file_info, 'error' => $e->getMessage()]);
                    continue;
                }
            }
        }


        // Initialize AI service
        $llm = createLLMInstance($chatConfig->getAiService());
        $llm->setPrompt($clean_system_prompt);
        $llm->setMaxMemoryMessages($chatConfig->getMaxMemory());
        
        // Convert messages to AI format (including attachments)
        $aiMessages = [];
        foreach ($recentMessages as $msg) {
            $content = $msg->getMessage();
            $attachments = $msg->getAttachments();
            
            // If message has attachments, create multimodal content
            if (!empty($attachments)) {
                $multimodalContent = [];
                
                // Add text content if not empty
                if (!empty(trim($content))) {
                    $multimodalContent[] = [
                        'type' => 'text',
                        'text' => $content
                    ];
                }
                
                // Add image and PDF attachments
                foreach ($attachments as $attachment) {
                    try {
                        if ($attachment->isImage()) {
                            $imageData = $attachment->getOptimizedContentAsBase64();
                            if ($imageData) {
                                $multimodalContent[] = [
                                    'type' => 'image_url',
                                    'image_url' => [
                                        'url' => $imageData
                                    ]
                                ];
                            }
                        } elseif ($attachment->isPdf()) {
                            // Handle PDF attachments - convert to image pages
                            $pdfDataUrls = $attachment->getDataUrl(); // Returns array of page URLs
                            if ($pdfDataUrls && is_array($pdfDataUrls)) {
                                foreach ($pdfDataUrls as $pageDataUrl) {
                                    if ($pageDataUrl) {
                                        $multimodalContent[] = [
                                            'type' => 'image_url',
                                            'image_url' => [
                                                'url' => $pageDataUrl
                                            ]
                                        ];
                                    }
                                }
                            } else {
                                $logger->warning("PDF attachment conversion failed", ['attachment_id' => $attachment->getAttachmentId()]);
                            }
                        }
                    } catch (\Exception $e) {
                        $logger->warning("Attachment processing error", ['attachment_id' => $attachment->getAttachmentId(), 'error' => $e->getMessage()]);
                    }
                }
                
                $aiMessages[] = [
                    'role' => $msg->getRole(),
                    'content' => $multimodalContent
                ];
            } else {
                // Text-only message
                $aiMessages[] = [
                    'role' => $msg->getRole(),
                    'content' => $content
                ];
            }
        }
        
        // Send to AI
        $aiResponse = $llm->sendMessagesArray($aiMessages, $contextResources);
        
        // Add AI response to session
        $aiMessage = $session->addMessage('assistant', $aiResponse);
        
        // 11. Return chat state
        return [
            'success' => true,
            'message' => $aiResponse,
            'session' => $session->toArray(),
            'messages' => array_map(fn($msg) => $msg->toArray(), $session->getMessages()),
            'experimental' => true,
            'context_resources_count' => count($contextResources)
        ];
        
    } catch (\Exception $e) {
        $logger->error("Message sending failed", [
            'chat_id' => $chat_id,
            'error' => $e->getMessage(),
            'trace' => $e->getTraceAsString()
        ]);
        // Return detailed error for debugging
        return [
            'error' => 'Failed to send message: ' . $e->getMessage(),
            'debug' => [
                'exception' => get_class($e),
                'file' => $e->getFile(),
                'line' => $e->getLine()
            ]
        ];
    }
}

/**
 * Handle send message with Server-Sent Events streaming
 */
function handleSendMessageStream(array $data): void
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    // Set Server-Sent Events headers
    header('Content-Type: text/event-stream');
    header('Cache-Control: no-cache');
    header('Connection: keep-alive');
    
    // Extract required data from frontend
    $chat_id = $data['chat_id'] ?? '';
    $user_message = $data['message'] ?? '';
    
    // Handle attachment_ids - could be JSON string from GET parameters or array from POST
    $attachment_ids = $data['attachment_ids'] ?? [];
    if (is_string($attachment_ids) && !empty($attachment_ids)) {
        $attachment_ids = json_decode($attachment_ids, true) ?: [];
    }
    if (!is_array($attachment_ids)) {
        $attachment_ids = [];
    }
    
    if (empty($chat_id) || empty($user_message)) {
        echo "data: " . json_encode(['error' => 'Missing required parameters']) . "\n\n";
        exit;
    }
    
    // Validate message length
    $char_limit = (int)(\platform\AIChatPageComponentConfig::get('characters_limit') ?: 2000);
    if (strlen($user_message) > $char_limit) {
        echo "data: " . json_encode(['error' => 'Message too long']) . "\n\n";
        exit;
    }
    
    $user_id = $DIC->user()->getId();
    
    try {
        // Load chat configuration 
        $chatConfig = new ChatConfig($chat_id);
        if (!$chatConfig->exists()) {
            echo "data: " . json_encode(['error' => 'Chat configuration not found']) . "\n\n";
            exit;
        }
        
        // Get or create session
        $session = ChatSession::getOrCreateForUserAndChat($user_id, $chat_id);
        
        // Add user message to session
        $userMessage = $session->addMessage('user', $user_message);
        
        // Process attachments if provided
        if (!empty($attachment_ids)) {
            foreach ($attachment_ids as $attachment_id) {
                if (is_numeric($attachment_id)) {
                    $userMessage->addAttachment($attachment_id);
                }
            }
        }
        
        // Get recent messages for context
        $recent_limit = min($chatConfig->getMaxMemory(), 20);
        $recentMessages = $session->getRecentMessages($recent_limit);
        
        // Build context resources (background files, page context, etc.)
        $contextResources = [];
        
        // Add page context if enabled
        if ($chatConfig->isIncludePageContext()) {
            $pageContext = getPageContextForChat($chatConfig);
            if (!empty($pageContext)) {
                $contextResources[] = [
                    'kind' => 'page_context',
                    'title' => 'Page Context',
                    'content' => $pageContext,
                    'mime_type' => 'text/plain'
                ];
            }
        }
        
        // Add background files as resources 
        $background_files = $chatConfig->getBackgroundFiles();
        if (!empty($background_files)) {
            $irss = $DIC->resourceStorage();
            
            foreach ($background_files as $file_id) {
                try {
                    $identification = $irss->manage()->find($file_id);
                    if ($identification === null) continue;
                    
                    $revision = $irss->manage()->getCurrentRevision($identification);
                    if ($revision === null) continue;
                    
                    $suffix = strtolower($revision->getInformation()->getSuffix());
                    $mime_type = $revision->getInformation()->getMimeType();
                    
                    // Process ALL file types as structured resources
                    if (in_array($suffix, ['txt', 'md', 'csv'])) {
                        // Text files
                        $content = extractFileContentFromIRSS($identification);
                        if (!empty($content)) {
                            $contextResources[] = [
                                'kind' => 'text_file',
                                'id' => 'bg-text-' . $file_id,
                                'title' => $revision->getTitle(),
                                'mime_type' => $mime_type,
                                'content' => $content
                            ];
                        }
                    } elseif (in_array($suffix, ['jpg', 'jpeg', 'png', 'gif', 'webp'])) {
                        // Single images
                        $content = extractFileContentFromIRSS($identification);
                        if (!empty($content)) {
                            $contextResources[] = [
                                'kind' => 'image_file',
                                'id' => 'bg-img-' . $file_id,
                                'title' => $revision->getTitle(),
                                'mime_type' => $mime_type,
                                'url' => $content
                            ];
                        }
                    } elseif ($suffix === 'pdf') {
                        // PDF pages (converted to images)
                        $content = extractFileContentFromIRSS($identification);
                        if (!empty($content) && is_array($content)) {
                            foreach ($content as $pageIndex => $pageDataUrl) {
                                if (!empty($pageDataUrl)) {
                                    $contextResources[] = [
                                        'kind' => 'pdf_page',
                                        'id' => 'bg-pdf-' . $file_id . '-p' . ($pageIndex + 1),
                                        'title' => $revision->getTitle() . ' (Page ' . ($pageIndex + 1) . ')',
                                        'mime_type' => 'image/png',
                                        'page_number' => $pageIndex + 1,
                                        'source_file' => $revision->getTitle(),
                                        'url' => $pageDataUrl
                                    ];
                                }
                            }
                        }
                    }
                } catch (\Exception $e) {
                    $logger->warning("Background file processing failed", ['file_id' => $file_id, 'error' => $e->getMessage()]);
                    continue;
                }
            }
        }
        
        // Initialize AI service with streaming enabled
        $llm = createLLMInstance($chatConfig->getAiService());
        $llm->setStreaming(true); // Enable streaming
        $llm->setPrompt($chatConfig->getSystemPrompt());
        $llm->setMaxMemoryMessages($chatConfig->getMaxMemory());
        
        // Send start event
        echo "data: " . json_encode(['type' => 'start']) . "\n\n";
        flush();
        
        // Convert messages to AI format
        $aiMessages = [];
        foreach ($recentMessages as $msg) {
            $content = $msg->getMessage();
            $attachments = $msg->getAttachments();
            
            if (!empty($attachments)) {
                $multimodalContent = [];
                if (!empty(trim($content))) {
                    $multimodalContent[] = ['type' => 'text', 'text' => $content];
                }
                
                foreach ($attachments as $attachment) {
                    try {
                        if ($attachment->isImage()) {
                            $imageData = $attachment->getOptimizedContentAsBase64();
                            if ($imageData) {
                                $multimodalContent[] = [
                                    'type' => 'image_url',
                                    'image_url' => ['url' => $imageData]
                                ];
                            }
                        } elseif ($attachment->isPdf()) {
                            $pdfDataUrls = $attachment->getDataUrl();
                            if ($pdfDataUrls && is_array($pdfDataUrls)) {
                                foreach ($pdfDataUrls as $pageDataUrl) {
                                    if ($pageDataUrl) {
                                        $multimodalContent[] = [
                                            'type' => 'image_url',
                                            'image_url' => ['url' => $pageDataUrl]
                                        ];
                                    }
                                }
                            }
                        }
                    } catch (\Exception $e) {
                        $logger->warning("Attachment processing failed", ['error' => $e->getMessage()]);
                    }
                }
                
                $aiMessages[] = [
                    'role' => $msg->getRole(),
                    'content' => $multimodalContent
                ];
            } else {
                $aiMessages[] = [
                    'role' => $msg->getRole(),
                    'content' => $content
                ];
            }
        }
        
        // Send to AI with streaming (this will output chunks directly)
        $aiResponse = $llm->sendMessagesArray($aiMessages, $contextResources);
        
        // Send completion event
        echo "data: " . json_encode(['type' => 'complete', 'message' => $aiResponse]) . "\n\n";
        flush();
        
        // Add AI response to session
        $session->addMessage('assistant', $aiResponse);
        
    } catch (\Exception $e) {
        $logger->error("Streaming message error: " . $e->getMessage());
        echo "data: " . json_encode(['error' => 'Failed to send message']) . "\n\n";
    }
    
    exit;
}

/**
 * Get page context for chat configuration
 */
function getPageContextForChat(ChatConfig $chatConfig): string
{
    global $DIC;
    $logger = $DIC->logger()->root();

    $page_id     = (int) $chatConfig->getPageId();     // bei wpg: DIESE musst du nehmen
    $parent_id   = (int) $chatConfig->getParentId();   // Objekt-ID (z.B. Kurs, Wiki-Objekt)
    $parent_type = (string) $chatConfig->getParentType();

    // COPage-Typ bestimmen
    $copage_type_map = [
        'crs' => 'cont',
        'grp' => 'cont',
        'cont'=> 'cont',
        'cat' => 'cont',
        'lm'  => 'lm',
        'wpg' => 'wpg',  // Wiki-Page (einzelne Seite)
        'wiki'=> 'wpg',  // falls du hier "wiki" speicherst, auch auf wpg mappen
        'copa'=> 'copa',
        'glo' => 'glo',
        'blp' => 'blp',
        'frm' => 'frm',
        'tst' => 'tst',
        'qpl' => 'qpl'
    ];
    $copage_type = $copage_type_map[$parent_type] ?? null;
    if (!$copage_type) {
        $logger->info("No page context available for parent type", ['parent_type' => $parent_type]);
        return '';
    }

    $page_manager = $DIC->copage()->internal()->domain()->page();
    $page = null;

    try {
        if ($copage_type === 'wpg') {
            if ($page_id > 0) {
                $page = $page_manager->get('wpg', $page_id);
            } else {
                $logger->info("Missing wiki page_id for wpg context");
                return '';
            }
        } else {
            $target_page_id = $page_id ?: $parent_id;
            if ($target_page_id <= 0) {
                $logger->info("No page/parent id available", ['copage_type' => $copage_type]);
                return '';
            }
            $page = $page_manager->get($copage_type, $target_page_id);
        }
    } catch (\Throwable $e) {
        $logger->error("page()->get() failed: " . $e->getMessage());
        $page = null;
    }

    if (!$page) {
        $logger->info("Service returned null for page context", ['copage_type' => $copage_type, 'id' => ($copage_type === 'wpg' ? $page_id : ($page_id ?: $parent_id))]);
        return '';
    }

    $content = $page->getRenderedContent() ?? '';
    $title = ilObject::_lookupTitle($parent_id);
    $description = ilObject::_lookupDescription($parent_id);

    $result = '';
    if ($title !== '')       { $result .= "Page Title: $title\n\n"; }
    if ($description !== '') { $result .= "Page Description: $description\n\n"; }
    if ($content !== '')     { $result .= "Page Content: $content\n\n"; }

    // Apply page context character limit to prevent token overflow
    $max_context_chars_config = \platform\AIChatPageComponentConfig::get('max_page_context_chars');
    $max_context_chars = $max_context_chars_config ? (int)$max_context_chars_config : 50000;
    
    // Log config source
    if ($max_context_chars_config !== null) {
    } else {
    }
    
    if (strlen($result) > $max_context_chars) {
        $original_length = strlen($result);
        $result = substr($result, 0, $max_context_chars) . "\n\n[Content truncated due to length limit]";
        $logger->info("Page context truncated", [
            'original_length' => $original_length,
            'truncated_length' => strlen($result) - 42, // minus truncation message
            'limit' => $max_context_chars,
            'truncated_chars' => $original_length - $max_context_chars
        ]);
    }

    $logger->debug("Return content, len=" . strlen($result) . ", type=" . $copage_type);
    return $result;
}

/**
 * Get background files context for chat configuration (TEXT FILES ONLY for system prompt)
 */
function getBackgroundFilesContextForChat(ChatConfig $chatConfig): string
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    try {
        $background_files = $chatConfig->getBackgroundFiles();
        if (empty($background_files)) {
            return '';
        }
        
        $context_parts = [];
        $irss = $DIC->resourceStorage();
        
        foreach ($background_files as $file_id) {
            try {
                $identification = $irss->manage()->find($file_id);
                if ($identification === null) {
                    continue;
                }
                
                $revision = $irss->manage()->getCurrentRevision($identification);
                if ($revision === null) {
                    continue;
                }
                
                // Only process text files for system prompt
                $suffix = strtolower($revision->getInformation()->getSuffix());
                if (!in_array($suffix, ['txt', 'md', 'csv'])) {
                    continue; // Skip non-text files
                }
                
                // Add file metadata
                $file_context = "File: " . $revision->getTitle();
                $file_context .= " (Type: " . $revision->getInformation()->getMimeType() . ")";
                $file_context .= "\n";
                
                // Extract content based on file type
                $content = extractFileContentFromIRSS($identification);
                if (!empty($content)) {
                    $file_context .= "Content: " . $content;
                }
                
                $context_parts[] = $file_context;
                
            } catch (\Exception $e) {
                $logger->warning("Background file context error", ['file_id' => $file_info['file_id'], 'error' => $e->getMessage()]);
                continue;
            }
        }
        
        return implode("\n\n---\n\n", $context_parts);
        
    } catch (\Exception $e) {
        $logger->error("Failed to build background files context", ['error' => $e->getMessage()]);
        return '';
    }
}

/**
 * Get background image messages for chat configuration (IMAGES and PDFs as multimodal messages)
 */
function getBackgroundImageMessagesForChat(ChatConfig $chatConfig): array
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    try {
        $background_files = $chatConfig->getBackgroundFiles();
        if (empty($background_files)) {
            return [];
        }
        
        $image_files = [];
        $irss = $DIC->resourceStorage();
        
        foreach ($background_files as $file_id) {
            try {
                $identification = $irss->manage()->find($file_id);
                if ($identification === null) {
                    continue;
                }
                
                $revision = $irss->manage()->getCurrentRevision($identification);
                if ($revision === null) {
                    continue;
                }
                
                // Process images and PDFs
                $suffix = strtolower($revision->getInformation()->getSuffix());
                $mime_type = $revision->getInformation()->getMimeType();
                
                if (in_array($suffix, ['jpg', 'jpeg', 'png', 'gif', 'webp']) || $suffix === 'pdf') {
                    $content = extractFileContentFromIRSS($identification);
                    if (!empty($content)) {
                        if ($suffix === 'pdf') {
                            // PDF content is already converted to image URLs by extractFileContentFromIRSS
                            if (is_array($content)) {
                                // Multiple PDF pages
                                foreach ($content as $page_data_url) {
                                    $image_files[] = [
                                        'type' => 'image_url',
                                        'image_url' => [
                                            'url' => $page_data_url
                                        ]
                                    ];
                                }
                            } else {
                                // Single page or data URL
                                $image_files[] = [
                                    'type' => 'image_url',
                                    'image_url' => [
                                        'url' => $content
                                    ]
                                ];
                            }
                        } else {
                            // Regular image
                            $image_files[] = [
                                'type' => 'image_url',
                                'image_url' => [
                                    'url' => $content
                                ]
                            ];
                        }
                    }
                }
                
            } catch (\Exception $e) {
                $logger->warning("Background image processing failed", ['file_id' => $file_info['file_id'], 'error' => $e->getMessage()]);
                continue;
            }
        }
        
        // Return as multimodal user message
        if (!empty($image_files)) {
            $content = [
                [
                    'type' => 'text',
                    'text' => 'Background Images: The following images have been uploaded as background context. Please analyze them and be ready to answer questions about their content:'
                ]
            ];
            
            $content = array_merge($content, $image_files);
            
            return [[
                'role' => 'user',
                'content' => $content
            ]];
        }
        
        return [];
        
    } catch (\Exception $e) {
        $logger->error("Failed to build background image messages", ['error' => $e->getMessage()]);
        return [];
    }
}

/**
 * Convert PDF to images directly using Ghostscript
 * Returns array of image data URLs for each page
 */
function convertPdfToImagesDirectly($identification, $revision, $irss): ?array
{
    global $DIC;
    $logger = $DIC->logger()->root();

    try {
        
        // Get PDF content from IRSS
        $stream = $irss->consume()->stream($identification);
        $pdf_content = $stream->getStream()->getContents();
        
        if (empty($pdf_content)) {
            $logger->info("PDF file is empty", ['title' => $revision->getTitle()]);
            return null;
        }
        
        // Create temporary file for PDF
        $temp_dir = sys_get_temp_dir();
        $temp_pdf = $temp_dir . '/aichat_pdf_' . uniqid() . '.pdf';
        $temp_png_pattern = $temp_dir . '/aichat_pdf_' . uniqid() . '_page_%03d.png';
        
        file_put_contents($temp_pdf, $pdf_content);
        
        // Use Ghostscript to convert PDF to PNG images
        $gs_command = "gs -dNOPAUSE -dBATCH -sDEVICE=png16m -dUseCropBox -r150 -sOutputFile=" . escapeshellarg($temp_png_pattern) . " " . escapeshellarg($temp_pdf) . " 2>&1";
        
        $output = shell_exec($gs_command);
        
        // Find generated PNG files
        $base_pattern = str_replace('%03d', '*', $temp_png_pattern);
        $png_files = glob($base_pattern);
        
        if (empty($png_files)) {
            @unlink($temp_pdf);
            $logger->warning("PDF conversion produced no images", ['title' => $revision->getTitle()]);
            return null;
        }
        
        // Convert to data URLs
        $data_urls = [];
        $max_pages = 20; // Limit to avoid overwhelming AI
        
        // Sort files to ensure correct page order
        sort($png_files);
        
        foreach ($png_files as $index => $png_file) {
            if ($index >= $max_pages) {
                break;
            }
            
            if (file_exists($png_file)) {
                $image_data = file_get_contents($png_file);
                if ($image_data !== false) {
                    // Optimize each page
                    require_once(__DIR__ . '/src/Service/ImageOptimizer.php');
                    $optimized = \ILIAS\Plugin\pcaic\Service\ImageOptimizer::optimize(
                        $image_data, 
                        'image/png'
                    );
                    
                    $base64Content = base64_encode($optimized['data']);
                    $data_url = 'data:' . $optimized['mime_type'] . ';base64,' . $base64Content;
                    $data_urls[] = $data_url;
                }
                @unlink($png_file); // Clean up temp file
            }
        }
        
        // Cleanup temp PDF
        @unlink($temp_pdf);

        return !empty($data_urls) ? $data_urls : null;
        
    } catch (\Exception $e) {
        $logger->error("PDF conversion failed", ['title' => $revision->getTitle(), 'error' => $e->getMessage()]);
        return null;
    }
}

/**
 * Extract content from IRSS file
 */
function extractFileContentFromIRSS($identification)
{
    global $DIC;
    $logger = $DIC->logger()->root();
    
    try {
        $irss = $DIC->resourceStorage();
        $revision = $irss->manage()->getCurrentRevision($identification);
        
        if ($revision === null) {
            return '';
        }
        
        $suffix = strtolower($revision->getInformation()->getSuffix());
        $stream = $irss->consume()->stream($identification);
        
        if ($stream === null) {
            return '';
        }
        
        $content = $stream->getStream()->getContents();
        
        switch ($suffix) {
            case 'txt':
            case 'md':
            case 'csv':
                return $content;
            case 'pdf':
                // Use direct Ghostscript conversion for background files
                return convertPdfToImagesDirectly($identification, $revision, $irss);
            case 'jpg':
            case 'jpeg':
            case 'png':
            case 'gif':
            case 'webp':
                // For images, return base64 data URL for AI analysis
                $imageContent = $stream->getStream()->getContents();
                $base64Image = base64_encode($imageContent);
                $mimeType = $revision->getInformation()->getMimeType();
                return "data:$mimeType;base64,$base64Image";
            default:
                return '[File: ' . $revision->getTitle() . ', Type: ' . $revision->getInformation()->getMimeType() . ']';
        }
    } catch (\Exception $e) {
        $logger->error("Failed to extract file content", ['error' => $e->getMessage()]);
        return '';
    }
}

/**
 * Update page context from current PageComponent context (backend only)
 * This handles moved/copied PageComponents by accessing the actual PageComponent XML
 */
function updatePageContextFromPageComponent(ChatConfig $chatConfig, string $chat_id): void
{
    global $DIC;
    $logger = $DIC->logger()->root();

    try {
        // Find the current PageComponent in ILIAS that contains this chat_id
        $currentPageInfo = findPageComponentByChat($chat_id);
        
        if ($currentPageInfo) {
            $current_parent_id = $chatConfig->getParentId();
            $current_parent_type = $chatConfig->getParentType();
            
            // Only update if different (PageComponent was moved/copied)
            if ($current_parent_id !== $currentPageInfo['parent_id'] || 
                $current_parent_type !== $currentPageInfo['parent_type']) {

                $logger->debug("Page context changed - old: parent_id=" . $current_parent_id . ", parent_type=" . $current_parent_type . " → new: parent_id=" . $currentPageInfo['parent_id'] . ", parent_type=" . $currentPageInfo['parent_type']);
                
                $chatConfig->setPageId($currentPageInfo['page_id'] ?: 0);
                $chatConfig->setParentId($currentPageInfo['parent_id']);
                $chatConfig->setParentType($currentPageInfo['parent_type']);
                $chatConfig->save();

                $logger->debug("Page context updated from actual PageComponent location");
            } else {
                $logger->debug("Page context already matches current PageComponent location");
            }
        } else {
            $logger->info("PageComponent not found for chat", ['chat_id' => $chat_id]);
        }
        
    } catch (\Exception $e) {
        $logger->warning("Failed to update page context", ['chat_id' => $chat_id, 'error' => $e->getMessage()]);
    }
}

/**
 * Find the PageComponent that contains a specific chat_id by searching ILIAS pages
 * This is backend-only and doesn't rely on frontend data
 */
function findPageComponentByChat(string $chat_id): ?array
{
    global $DIC;
    $logger = $DIC->logger()->root();

    try {
        // Search for PageComponents containing our chat_id in the copg_pobj_def table
        $db = $DIC->database();
        
        // Search for PageComponents of type 'pcaic' that might contain our chat_id
        $query = "SELECT pop.page_id, pop.parent_id, pop.parent_type 
                  FROM copg_pobj_def pop 
                  WHERE pop.class_name = 'pcaic' 
                  AND pop.component = 'pluginslot'";
        
        $result = $db->query($query);
        while ($row = $db->fetchAssoc($result)) {
            $page_id = (int)$row['page_id'];
            $parent_id = (int)$row['parent_id']; 
            $parent_type = $row['parent_type'];
            
            // Load the actual page content to check for our chat_id
            if (pageContentContainsChatId($page_id, $parent_id, $parent_type, $chat_id)) {
                return [
                    'page_id' => $page_id,
                    'parent_id' => $parent_id,
                    'parent_type' => $parent_type
                ];
            }
        }
        
        return null;
        
    } catch (\Exception $e) {
        $logger->error("Error finding PageComponent: " . $e->getMessage());
        return null;
    }
}

/**
 * Check if a specific page contains a chat_id in its PageComponent XML
 */
function pageContentContainsChatId(int $page_id, int $parent_id, string $parent_type, string $chat_id): bool
{
    global $DIC;
    
    try {
        // Map to COPage type
        $copage_type_map = [
            'crs' => 'cont', 'grp' => 'cont', 'lm' => 'lm',
            'wpg' => 'wpg', 'wiki' => 'wpg', 'glo' => 'glo',
            'blp' => 'blp', 'frm' => 'frm'
        ];
        
        $copage_type = $copage_type_map[$parent_type] ?? null;
        if (!$copage_type) {
            return false;
        }
        
        // Get the page content
        $page_manager = $DIC->copage()->internal()->domain()->page();
        $page = $page_manager->get($copage_type, $parent_id);
        
        if (!$page) {
            return false;
        }
        
        $page_xml = $page->getXMLContent();
        
        // Search for our chat_id in the XML content
        return strpos($page_xml, $chat_id) !== false;
        
    } catch (\Exception $e) {
        return false;
    }
}

/**
 * Create LLM instance
 */
function createLLMInstance(string $service)
{
    // Check if the requested service is enabled
    $available_services = \platform\AIChatPageComponentConfig::get('available_services') ?? [];
    
    switch ($service) {
        case 'openai':
            if (($available_services['openai'] ?? '0') !== '1') {
                throw new \Exception("OpenAI service is not enabled in plugin configuration.");
            }
            return \ai\AIChatPageComponentOpenAI::fromConfig();
            
        case 'ramses':
            if (($available_services['ramses'] ?? '1') !== '1') {
                throw new \Exception("RAMSES service is not enabled in plugin configuration.");
            }
            return \ai\AIChatPageComponentRAMSES::fromConfig();
            
        default:
            // Use the configured default service
            $default_service = \platform\AIChatPageComponentConfig::get('selected_ai_service') ?: 'ramses';
            if ($default_service !== $service) {
                return createLLMInstance($default_service);
            }
            
            // Fallback to RAMSES if nothing else works
            return \ai\AIChatPageComponentRAMSES::fromConfig();
    }
}


/**
 * Handle file upload
 */
function handleFileUpload(array $data): array
{
    global $DIC;
    $logger = $DIC->logger()->root();

    
    $chat_id = $data['chat_id'] ?? '';
    if (empty($chat_id)) {
        return ['error' => 'Missing chat_id parameter'];
    }
    
    // Check attachment limit (server-side validation)
    try {
        $chatSession = ChatSession::getOrCreateForUserAndChat($DIC->user()->getId(), $chat_id);
        $recentMessages = ChatMessage::getRecentForSession($chatSession->getSessionId(), 1);
        if (!empty($recentMessages)) {
            $latestMessage = $recentMessages[0];
            $existingAttachments = Attachment::getByMessageId($latestMessage->getMessageId());
            $maxAttachments = (int)(\platform\AIChatPageComponentConfig::get('max_attachments_per_message') ?: 5);
            
            if (count($existingAttachments) >= $maxAttachments) {
                return ['error' => "Maximum {$maxAttachments} attachments per message allowed"];
            }
        }
    } catch (\Exception $e) {
        // Log error but continue - this is just additional validation
        $logger->warning("Could not check attachment limit", ['error' => $e->getMessage()]);
    }
    
    // Check if file was uploaded
    if (empty($_FILES) || !isset($_FILES['file'])) {
        return ['error' => 'No file uploaded'];
    }
    
    $upload_info = $_FILES['file'];
    
    // Basic validation
    if ($upload_info['error'] !== UPLOAD_ERR_OK) {
        return ['error' => 'File upload failed'];
    }
    
    // Use new FileUploadValidator for comprehensive validation
    $validation_result = FileUploadValidator::validateUpload($upload_info, 'chat', $chat_id);
    if (!$validation_result['success']) {
        $logger->info("File upload validation failed", [
            'filename' => $upload_info['name'] ?? 'unknown',
            'size' => $upload_info['size'],
            'error' => $validation_result['error']
        ]);
        return ['error' => $validation_result['error']];
    }
    
    $logger->debug("File upload validation passed", [
        'filename' => $upload_info['name'] ?? 'unknown',
        'size' => $upload_info['size']
    ]);
    
    try {
        // Store file using ILIAS Resource Storage
        $resource_storage = $DIC->resourceStorage();
        $stakeholder = new \ILIAS\Plugin\pcaic\Storage\ResourceStakeholder();
        
        // Use ILIAS FileUpload service
        $upload_service = $DIC->upload();
        $upload_service->process();
        
        if (!$upload_service->hasUploads()) {
            throw new \Exception('No valid uploads found');
        }
        
        $upload_results = $upload_service->getResults();
        $upload_result = $upload_results[array_keys($upload_results)[0]];
        
        if (!$upload_result->isOK()) {
            throw new \Exception('Upload validation failed');
        }
        
        // Store in IRSS
        $resource_id = $resource_storage->manage()->upload($upload_result, $stakeholder);
        
        // Create attachment using proper Attachment model
        $attachment = new \ILIAS\Plugin\pcaic\Model\Attachment();
        $attachment->setMessageId(0); // Will be updated when message is sent
        $attachment->setChatId($chat_id);
        $attachment->setUserId($DIC->user()->getId());
        $attachment->setResourceId($resource_id->serialize());
        $attachment->setTimestamp(date('Y-m-d H:i:s'));
        $attachment->save();
        
        return [
            'success' => true,
            'attachment' => [
                'id' => $attachment->getId(),
                'title' => $attachment->getTitle(),
                'size' => $attachment->getSize(),
                'mime_type' => $attachment->getMimeType(),
                'is_image' => $attachment->isImage(),
                'download_url' => $attachment->getDownloadUrl(),
                'preview_url' => $attachment->getPreviewUrl()
            ]
        ];
        
    } catch (\Exception $e) {
        $logger->error("File upload failed", ['error' => $e->getMessage()]);
        return ['error' => 'File upload failed'];
    }
}

/**
 * Get upload configuration and restrictions for frontend
 * 
 * Provides information about allowed file types, size limits,
 * and whether uploads are enabled for different contexts.
 * 
 * @param array $data Request data
 * @return array Configuration response
 */
function getUploadConfig(array $data): array
{
    $upload_type = $data['upload_type'] ?? 'chat'; // 'chat' or 'background'
    
    try {
        // Get upload restrictions
        $is_enabled = FileUploadValidator::isUploadEnabled($upload_type);
        $allowed_extensions = FileUploadValidator::getAllowedExtensions($upload_type);
        
        // Get size limits from configuration
        $max_file_size_mb = \platform\AIChatPageComponentConfig::get('max_file_size_mb') ?? 5;
        $max_attachments = \platform\AIChatPageComponentConfig::get('max_attachments_per_message') ?? 5;
        $max_total_upload_mb = \platform\AIChatPageComponentConfig::get('max_total_upload_size_mb') ?? 25;
        
        // Build MIME type mapping for allowed extensions
        $extension_to_mime = [
            'pdf' => 'application/pdf',
            'txt' => 'text/plain',
            'md' => 'text/markdown',
            'csv' => 'text/csv',
            'jpg' => 'image/jpeg',
            'jpeg' => 'image/jpeg',
            'png' => 'image/png',
            'gif' => 'image/gif',
            'webp' => 'image/webp'
        ];
        
        $allowed_mime_types = [];
        foreach ($allowed_extensions as $ext) {
            if (isset($extension_to_mime[$ext])) {
                $allowed_mime_types[] = $extension_to_mime[$ext];
            }
        }
        
        return [
            'success' => true,
            'upload_enabled' => $is_enabled,
            'upload_type' => $upload_type,
            'allowed_extensions' => $allowed_extensions,
            'allowed_mime_types' => array_unique($allowed_mime_types),
            'max_file_size_mb' => $max_file_size_mb,
            'max_attachments_per_message' => $max_attachments,
            'max_total_upload_size_mb' => $max_total_upload_mb,
            'extensions_display' => implode(', ', $allowed_extensions)
        ];
        
    } catch (\Exception $e) {
        global $DIC;
        $DIC->logger()->comp('pcaic')->error('Failed to get upload config', [
            'error' => $e->getMessage(),
            'upload_type' => $upload_type
        ]);
        
        return [
            'success' => false,
            'error' => 'Failed to load upload configuration'
        ];
    }
}

/**
 * Get global configuration with administrator limits
 * 
 * Returns global configuration settings that may override local PageComponent
 * settings. Administrators can set system-wide limits that take precedence.
 * 
 * @param array $data Request data
 * @return array Configuration response
 */
function getGlobalConfig(array $data): array
{
    try {
        // Get upload configuration (existing functionality)
        $upload_config = getUploadConfig(['upload_type' => 'chat']);
        
        if (!$upload_config['success']) {
            return $upload_config; // Return error from upload config
        }
        
        // Get global chat limits from plugin configuration
        // Use the standard plugin settings as global limits (administrator-configured)
        $max_char_limit = \platform\AIChatPageComponentConfig::get('characters_limit');
        $max_memory_limit = \platform\AIChatPageComponentConfig::get('max_memory_messages');
        
        // Convert to proper integers, ensuring valid values
        $max_char_limit = $max_char_limit ? (int)$max_char_limit : null;
        $max_memory_limit = $max_memory_limit ? (int)$max_memory_limit : null;
        
        global $DIC;
        $DIC->logger()->comp('pcaic')->debug('Global limits from plugin configuration', [
            'characters_limit_from_config' => \platform\AIChatPageComponentConfig::get('characters_limit'),
            'max_memory_from_config' => \platform\AIChatPageComponentConfig::get('max_memory_messages'),
            'final_char_limit' => $max_char_limit,
            'final_memory_limit' => $max_memory_limit
        ]);
        
        // Combine upload configuration with chat limits
        $global_config = $upload_config;
        $global_config['max_char_limit'] = $max_char_limit ? (int)$max_char_limit : null;
        $global_config['max_memory_limit'] = $max_memory_limit ? (int)$max_memory_limit : null;
        
        global $DIC;
        $DIC->logger()->comp('pcaic')->debug('Global configuration requested', [
            'upload_enabled' => $global_config['upload_enabled'],
            'max_char_limit' => $global_config['max_char_limit'],
            'max_memory_limit' => $global_config['max_memory_limit'],
            'allowed_extensions_count' => count($global_config['allowed_extensions'])
        ]);
        
        return $global_config;
        
    } catch (\Exception $e) {
        global $DIC;
        $DIC->logger()->comp('pcaic')->error('Failed to get global config', [
            'error' => $e->getMessage()
        ]);
        
        return [
            'success' => false,
            'error' => 'Failed to load global configuration'
        ];
    }
}

/**
 * Parse boolean parameter
 */
function parseBooleanParam($value): bool
{
    if (is_bool($value)) {
        return $value;
    }
    
    if (is_string($value)) {
        return in_array(strtolower($value), ['true', '1', 'yes', 'on']);
    }
    
    return (bool)$value;
}

/**
 * Send API response
 */
function sendApiResponse($data, int $httpCode = 200): void
{
    // Clean ALL output buffers to prevent JSON corruption
    while (ob_get_level() > 0) {
        ob_end_clean();
    }

    // Prevent "headers already sent" errors by checking first
    if (!headers_sent()) {
        http_response_code($httpCode);
        header('Content-Type: application/json');
    }

    echo json_encode($data);
    exit();
}