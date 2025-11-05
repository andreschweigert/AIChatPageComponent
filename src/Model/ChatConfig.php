<?php

namespace ILIAS\Plugin\pcaic\Model;

/**
 * ChatConfig Model - PageComponent Configuration
 * 
 * Represents the configuration of a PageComponent (one per PageComponent)
 * Contains: system_prompt, ai_service, background_files, etc.
 *
 * @author Nadimo Staszak <nadimo.staszak@uni-koeln.de>
 */
class ChatConfig
{
    private string $chatId;
    private int $pageId = 0;
    private int $parentId = 0;
    private string $parentType = '';
    private string $title = '';
    private string $systemPrompt = '';
    private string $aiService = 'ramses';
    private int $maxMemory = 10;
    private int $charLimit = 2000;
    private array $backgroundFiles = [];
    private bool $persistent = true;
    private bool $includePageContext = true;
    private bool $enableChatUploads = false;
    private bool $enableStreaming = true;
    private string $disclaimer = '';
    private ?\DateTime $createdAt = null;
    private ?\DateTime $updatedAt = null;

    public function __construct(string $chatId = null)
    {
        if ($chatId) {
            $this->chatId = $chatId;
            $this->load();
        } else {
            $this->chatId = uniqid('chat_', true);
            $this->createdAt = new \DateTime();
            $this->updatedAt = new \DateTime();
            
            // Load defaults from global configuration for new instances
            $this->loadGlobalDefaults();
        }
    }

    /**
     * Load configuration from database
     */
    private function load(): bool
    {
        global $DIC;
        $db = $DIC->database();

        $query = "SELECT * FROM pcaic_chats WHERE chat_id = " . $db->quote($this->chatId, 'text');
        $result = $db->query($query);
        
        if ($row = $db->fetchAssoc($result)) {
            $this->pageId = (int)$row['page_id'];
            $this->parentId = (int)$row['parent_id'];
            $this->parentType = $row['parent_type'];
            $this->title = $row['title'] ?? '';
            $this->systemPrompt = $row['system_prompt'] ?? '';

            // Use global selected_ai_service as fallback if ai_service is not set
            $default_service = \platform\AIChatPageComponentConfig::get('selected_ai_service') ?: 'openai';
            $configured_service = $row['ai_service'] ?? $default_service;

            // Validate that the configured service is actually available
            $available_services = \platform\AIChatPageComponentConfig::get('available_services') ?? [];
            if (!empty($configured_service) && isset($available_services[$configured_service]) && $available_services[$configured_service] === '1') {
                $this->aiService = $configured_service;
            } else {
                // Fall back to default service if configured service is not available
                $this->aiService = $default_service;
            }

            $this->maxMemory = (int)$row['max_memory'];
            $this->charLimit = (int)$row['char_limit'];
            
            // Parse JSON background files
            $bgFiles = $row['background_files'];
            if (is_string($bgFiles)) {
                $this->backgroundFiles = json_decode($bgFiles, true) ?? [];
            } elseif (is_array($bgFiles)) {
                $this->backgroundFiles = $bgFiles;
            }
            
            $this->persistent = (bool)$row['persistent'];
            $this->includePageContext = (bool)$row['include_page_context'];
            $this->enableChatUploads = (bool)$row['enable_chat_uploads'];
            $this->enableStreaming = (bool)($row['enable_streaming'] ?? true);
            $this->disclaimer = $row['disclaimer'] ?? '';
            
            $this->createdAt = $row['created_at'] ? new \DateTime($row['created_at']) : null;
            $this->updatedAt = $row['updated_at'] ? new \DateTime($row['updated_at']) : null;
            
            return true;
        }
        
        return false;
    }

    /**
     * Load default values from global plugin configuration
     */
    private function loadGlobalDefaults(): void
    {
        try {
            require_once(__DIR__ . '/../../classes/platform/class.AIChatPageComponentConfig.php');
            
            // Load defaults from global configuration
            $default_prompt = \platform\AIChatPageComponentConfig::get('default_prompt');
            if (!empty($default_prompt)) {
                $this->systemPrompt = $default_prompt;
            }
            
            $default_disclaimer = \platform\AIChatPageComponentConfig::get('default_disclaimer');
            if (!empty($default_disclaimer)) {
                $this->disclaimer = $default_disclaimer;
            }
            
            $char_limit = \platform\AIChatPageComponentConfig::get('characters_limit');
            if (!empty($char_limit)) {
                $this->charLimit = (int)$char_limit;
            }
            
            $max_memory = \platform\AIChatPageComponentConfig::get('max_memory_messages');
            if (!empty($max_memory)) {
                $this->maxMemory = (int)$max_memory;
            }

            // Load selected AI service from global configuration
            $selected_service = \platform\AIChatPageComponentConfig::get('selected_ai_service');
            if (!empty($selected_service)) {
                $this->aiService = $selected_service;
            }

            global $DIC;
            $DIC->logger()->comp('pcaic')->debug("Loaded global defaults for new ChatConfig", [
                'system_prompt_length' => strlen($this->systemPrompt),
                'disclaimer_length' => strlen($this->disclaimer),
                'char_limit' => $this->charLimit,
                'max_memory' => $this->maxMemory,
                'ai_service' => $this->aiService
            ]);
            
        } catch (\Exception $e) {
            global $DIC;
            $DIC->logger()->comp('pcaic')->warning("Failed to load global defaults for ChatConfig", [
                'error' => $e->getMessage()
            ]);
        }
    }

    /**
     * Save configuration to database
     */
    public function save(): bool
    {
        global $DIC;
        $db = $DIC->database();

        $this->updatedAt = new \DateTime();

        // Check if exists
        $query = "SELECT chat_id FROM pcaic_chats WHERE chat_id = " . $db->quote($this->chatId, 'text');
        $result = $db->query($query);
        $exists = $db->fetchAssoc($result);

        $values = [
            'page_id' => ['integer', $this->pageId],
            'parent_id' => ['integer', $this->parentId],
            'parent_type' => ['text', $this->parentType],
            'title' => ['text', $this->title],
            'system_prompt' => ['clob', $this->systemPrompt],
            'ai_service' => ['text', $this->aiService],
            'max_memory' => ['integer', $this->maxMemory],
            'char_limit' => ['integer', $this->charLimit],
            'background_files' => ['clob', json_encode($this->backgroundFiles)],
            'persistent' => ['integer', $this->persistent ? 1 : 0],
            'include_page_context' => ['integer', $this->includePageContext ? 1 : 0],
            'enable_chat_uploads' => ['integer', $this->enableChatUploads ? 1 : 0],
            'enable_streaming' => ['integer', $this->enableStreaming ? 1 : 0],
            'disclaimer' => ['clob', $this->disclaimer],
            'updated_at' => ['timestamp', $this->updatedAt->format('Y-m-d H:i:s')]
        ];

        if ($exists) {
            // Update
            $db->update('pcaic_chats', $values, ['chat_id' => ['text', $this->chatId]]);
        } else {
            // Insert - initialize createdAt if not set
            if (!$this->createdAt) {
                $this->createdAt = new \DateTime();
            }
            $values['chat_id'] = ['text', $this->chatId];
            $values['created_at'] = ['timestamp', $this->createdAt->format('Y-m-d H:i:s')];
            $db->insert('pcaic_chats', $values);
        }

        return true;
    }

    /**
     * Delete configuration
     */
    public function delete(): bool
    {
        global $DIC;
        $db = $DIC->database();

        // Delete will cascade to sessions and messages due to foreign keys
        $query = "DELETE FROM pcaic_chats WHERE chat_id = " . $db->quote($this->chatId, 'text');
        $db->manipulate($query);

        return true;
    }

    /**
     * Check if configuration exists
     */
    public function exists(): bool
    {
        global $DIC;
        $db = $DIC->database();

        $query = "SELECT chat_id FROM pcaic_chats WHERE chat_id = " . $db->quote($this->chatId, 'text');
        $result = $db->query($query);
        return $db->fetchAssoc($result) !== false;
    }

    /**
     * Get all active sessions for this chat
     */
    public function getSessions(): array
    {
        global $DIC;
        $db = $DIC->database();

        $query = "SELECT session_id FROM pcaic_sessions WHERE chat_id = " . $db->quote($this->chatId, 'text') . " AND is_active = 1";
        $result = $db->query($query);
        
        $sessions = [];
        while ($row = $db->fetchAssoc($result)) {
            $sessions[] = new ChatSession($row['session_id']);
        }
        
        return $sessions;
    }

    // Getters and Setters
    public function getChatId(): string { return $this->chatId; }
    public function setChatId(string $chatId): void { $this->chatId = $chatId; }
    public function getPageId(): int { return $this->pageId; }
    public function setPageId(int $pageId): void { $this->pageId = $pageId; }
    public function getParentId(): int { return $this->parentId; }
    public function setParentId(int $parentId): void { $this->parentId = $parentId; }
    public function getParentType(): string { return $this->parentType; }
    public function setParentType(string $parentType): void { $this->parentType = $parentType; }
    public function getTitle(): string { return $this->title; }
    public function setTitle(string $title): void { $this->title = $title; }
    public function getSystemPrompt(): string { return $this->systemPrompt; }
    public function setSystemPrompt(string $systemPrompt): void { $this->systemPrompt = $systemPrompt; }
    public function getAiService(): string { return $this->aiService; }
    public function setAiService(string $aiService): void { $this->aiService = $aiService; }
    public function getMaxMemory(): int { return $this->maxMemory; }
    public function setMaxMemory(int $maxMemory): void { $this->maxMemory = $maxMemory; }
    public function getCharLimit(): int { return $this->charLimit; }
    public function setCharLimit(int $charLimit): void { $this->charLimit = $charLimit; }
    public function getBackgroundFiles(): array { return $this->backgroundFiles; }
    public function setBackgroundFiles(array $backgroundFiles): void { $this->backgroundFiles = $backgroundFiles; }
    public function isPersistent(): bool { return $this->persistent; }
    public function setPersistent(bool $persistent): void { $this->persistent = $persistent; }
    public function isIncludePageContext(): bool { return $this->includePageContext; }
    public function setIncludePageContext(bool $includePageContext): void { $this->includePageContext = $includePageContext; }
    public function isEnableChatUploads(): bool { return $this->enableChatUploads; }
    public function setEnableChatUploads(bool $enableChatUploads): void { $this->enableChatUploads = $enableChatUploads; }
    public function isEnableStreaming(): bool { return $this->enableStreaming; }
    public function setEnableStreaming(bool $enableStreaming): void { $this->enableStreaming = $enableStreaming; }
    public function getDisclaimer(): string { return $this->disclaimer; }
    public function setDisclaimer(string $disclaimer): void { $this->disclaimer = $disclaimer; }
    public function getCreatedAt(): ?\DateTime { return $this->createdAt; }
    public function getUpdatedAt(): ?\DateTime { return $this->updatedAt; }

    /**
     * Convert to array for API responses
     */
    public function toArray(): array
    {
        return [
            'chat_id' => $this->chatId,
            'page_id' => $this->pageId,
            'parent_id' => $this->parentId,
            'parent_type' => $this->parentType,
            'title' => $this->title,
            'system_prompt' => $this->systemPrompt,
            'ai_service' => $this->aiService,
            'max_memory' => $this->maxMemory,
            'char_limit' => $this->charLimit,
            'background_files' => $this->backgroundFiles,
            'persistent' => $this->persistent,
            'include_page_context' => $this->includePageContext,
            'enable_chat_uploads' => $this->enableChatUploads,
            'disclaimer' => $this->disclaimer,
            'created_at' => $this->createdAt?->format('Y-m-d H:i:s'),
            'updated_at' => $this->updatedAt?->format('Y-m-d H:i:s')
        ];
    }
}