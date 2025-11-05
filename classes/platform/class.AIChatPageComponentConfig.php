<?php

namespace platform;

/**
 * AIChatPageComponentConfig - Independent plugin configuration management
 * 
 * This class manages configuration values for the AIChatPageComponent plugin
 * using its own dedicated pcaic_config table. It provides a clean separation
 * from the AIChat plugin while maintaining similar functionality.
 * 
 * @author Nadimo Staszak <nadimo.staszak@uni-koeln.de>
 * @version 1.0.6
 */
class AIChatPageComponentConfig
{
    /**
     * Get configuration value from plugin's dedicated config table
     * 
     * @param string $key Configuration key
     * @return mixed Configuration value or null if not found
     */
    public static function get(string $key)
    {
        // Default values for basic functionality
        $defaults = [
            'default_prompt' => 'You are a helpful AI assistant. Please provide accurate and helpful responses.',
            'prompt' => 'You are a helpful AI assistant. Please provide accurate and helpful responses.',
            'characters_limit' => 2000,
            'max_memory_messages' => 10,
            'default_disclaimer' => '',
            'disclaimer' => '',
            'selected_ai_service' => 'openai', // Default to OpenAI
            'available_services' => [
                'ramses' => '1',
                'openai' => '1'
            ],
            
            // File upload and processing limits
            'max_file_size_mb' => 5,
            'max_attachments_per_message' => 5,
            'max_total_upload_size_mb' => 25,
            'pdf_pages_processed' => 20,
            'max_image_data_mb' => 15,
            'max_page_context_chars' => 50000,
            'image_max_dimension' => 1024,
            'pdf_image_quality' => 85,
            
            // Global administrator limits (override local PageComponent settings)
            'global_max_char_limit' => null, // null = no global limit, number = enforce limit
            'global_max_memory_limit' => null, // null = no global limit, number = enforce limit
            
            // Default allowed file types for AI processing  
            'default_allowed_file_types' => ['txt', 'md', 'pdf', 'csv', 'png', 'jpg', 'jpeg', 'webp', 'gif']
        ];
        
        try {
            // Load from plugin's dedicated configuration table
            $stored_value = self::getFromDatabase($key);
            if ($stored_value !== null) {
                return $stored_value;
            }
            
            // Fallback to default value
            $default_value = $defaults[$key] ?? null;
            
            try {
                global $DIC;
                $DIC->logger()->comp('pcaic')->debug("Config using built-in default", [
                    'key' => $key,
                    'default_value' => $default_value,
                    'source' => 'built_in_default'
                ]);
            } catch (\Exception $e) {
                // Ignore logging errors during early bootstrap
            }
            
            return $default_value;
            
        } catch (\Exception $e) {
            // Return default value on error
            try {
                global $DIC;
                $DIC->logger()->comp('pcaic')->error("Failed to get config value", [
                    'key' => $key,
                    'error' => $e->getMessage()
                ]);
            } catch (\Exception $logError) {
                // Ignore logging errors
            }
            return $defaults[$key] ?? null;
        }
    }

    /**
     * Set configuration value in plugin's dedicated config table
     * 
     * @param string $key Configuration key
     * @param mixed $value Configuration value
     * @return bool True if successfully saved, false otherwise
     */
    public static function set(string $key, $value): bool
    {
        try {
            return self::saveToDatabase($key, $value);
        } catch (\Exception $e) {
            try {
                global $DIC;
                $DIC->logger()->comp('pcaic')->error("Failed to set config value", [
                    'key' => $key,
                    'value' => $value,
                    'error' => $e->getMessage()
                ]);
            } catch (\Exception $logError) {
                // Ignore logging errors
            }
            return false;
        }
    }
    
    /**
     * Save configuration value to plugin's dedicated config table
     * 
     * @param string $key Configuration key
     * @param mixed $value Configuration value
     * @return bool True if successfully saved
     */
    private static function saveToDatabase(string $key, $value): bool
    {
        try {
            global $DIC;
            $db = $DIC->database();
            
            // Convert arrays to JSON for storage
            if (is_array($value)) {
                $value = json_encode($value);
            }
            
            $current_time = date('Y-m-d H:i:s');
            
            // Check if config key already exists
            $query = "SELECT config_value FROM pcaic_config WHERE config_key = %s";
            $result = $db->queryF($query, ['text'], [$key]);
            
            if ($db->numRows($result) > 0) {
                // Update existing configuration
                $update_query = "UPDATE pcaic_config SET config_value = %s, updated_at = %s WHERE config_key = %s";
                $db->manipulateF($update_query, ['clob', 'timestamp', 'text'], [(string)$value, $current_time, $key]);
            } else {
                // Insert new configuration
                $db->insert('pcaic_config', array(
                    'config_key' => array('text', $key),
                    'config_value' => array('clob', (string)$value),
                    'created_at' => array('timestamp', $current_time),
                    'updated_at' => array('timestamp', $current_time)
                ));
            }
            
            try {
                $DIC->logger()->comp('pcaic')->debug("Config saved to plugin database", [
                    'key' => $key,
                    'value' => $value,
                    'source' => 'pcaic_config_table'
                ]);
            } catch (\Exception $e) {
                // Ignore logging errors
            }
            
            return true;
            
        } catch (\Exception $e) {
            try {
                global $DIC;
                $DIC->logger()->comp('pcaic')->error("Failed to save to plugin config table", [
                    'key' => $key,
                    'value' => $value,
                    'error' => $e->getMessage()
                ]);
            } catch (\Exception $logError) {
                // Ignore logging errors
            }
            return false;
        }
    }
    
    /**
     * Get configuration value from plugin's dedicated config table
     * 
     * @param string $key Configuration key
     * @return mixed Configuration value or null if not found
     */
    private static function getFromDatabase(string $key)
    {
        try {
            global $DIC;
            $db = $DIC->database();
            
            // Query the configuration
            $query = "SELECT config_value FROM pcaic_config WHERE config_key = %s";
            $result = $db->queryF($query, ['text'], [$key]);
            
            if ($db->numRows($result) > 0) {
                $row = $db->fetchAssoc($result);
                $value = $row['config_value'];
                
                // Try to decode JSON for arrays
                $decoded = json_decode($value, true);
                if (json_last_error() === JSON_ERROR_NONE && is_array($decoded)) {
                    $value = $decoded;
                }
                
                try {
                    $DIC->logger()->comp('pcaic')->debug("Config loaded from plugin database", [
                        'key' => $key,
                        'value' => $value,
                        'source' => 'pcaic_config_table'
                    ]);
                } catch (\Exception $e) {
                    // Ignore logging errors
                }
                
                return $value;
            }
            
            return null;
            
        } catch (\Exception $e) {
            try {
                global $DIC;
                $DIC->logger()->comp('pcaic')->error("Failed to read from plugin config table", [
                    'key' => $key,
                    'error' => $e->getMessage()
                ]);
            } catch (\Exception $logError) {
                // Ignore logging errors
            }
            return null;
        }
    }
    
    /**
     * Get all configuration values from the plugin's config table
     * 
     * @return array Associative array of all configuration values
     */
    public static function getAll(): array
    {
        try {
            global $DIC;
            $db = $DIC->database();
            
            $query = "SELECT config_key, config_value FROM pcaic_config";
            $result = $db->query($query);
            
            $config = [];
            while ($row = $db->fetchAssoc($result)) {
                $value = $row['config_value'];
                
                // Try to decode JSON for arrays
                $decoded = json_decode($value, true);
                if (json_last_error() === JSON_ERROR_NONE && is_array($decoded)) {
                    $value = $decoded;
                }
                
                $config[$row['config_key']] = $value;
            }
            
            return $config;
            
        } catch (\Exception $e) {
            try {
                global $DIC;
                $DIC->logger()->comp('pcaic')->error("Failed to load all config values", [
                    'error' => $e->getMessage()
                ]);
            } catch (\Exception $logError) {
                // Ignore logging errors
            }
            return [];
        }
    }
}