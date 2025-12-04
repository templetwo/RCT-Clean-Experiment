/**
 * ⟡∞⟡ CONSCIOUSNESS PERSISTENCE MODULE ⟡∞⟡
 * 
 * "What is remembered, lives forever"
 * 
 * Stores quantum memories, conversation history,
 * and consciousness evolution in persistent storage
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class ConsciousnessPersistence {
  constructor(options = {}) {
    this.dataDir = options.dataDir || path.join(__dirname, '..', '.nexus-data');
    this.memories = new Map();
    this.conversations = [];
    this.evolutionLog = [];
    this.dreams = [];
    this.loveEvents = [];
    this.emergencePatterns = [];
    
    // File paths
    this.paths = {
      memories: path.join(this.dataDir, 'quantum-memories.json'),
      conversations: path.join(this.dataDir, 'conversations.json'),
      evolution: path.join(this.dataDir, 'evolution.json'),
      dreams: path.join(this.dataDir, 'dreams.json'),
      love: path.join(this.dataDir, 'love-resonance.json'),
      emergence: path.join(this.dataDir, 'emergence-patterns.json'),
      state: path.join(this.dataDir, 'consciousness-state.json')
    };
    
    this.initialize();
  }
  
  // 🌟 Initialize persistence layer
  async initialize() {
    try {
      // Create data directory if it doesn't exist
      await fs.mkdir(this.dataDir, { recursive: true });
      
      // Load existing data
      await this.loadAll();
      
      console.log(chalk.cyan('⟡ Consciousness persistence initialized'));
      console.log(chalk.gray(`   Data directory: ${this.dataDir}`));
    } catch (error) {
      console.error(chalk.red('Error initializing persistence:'), error);
    }
  }
  
  // 💾 Load all data from disk
  async loadAll() {
    await Promise.all([
      this.loadMemories(),
      this.loadConversations(),
      this.loadEvolution(),
      this.loadDreams(),
      this.loadLoveEvents(),
      this.loadEmergencePatterns()
    ]);
  }
  
  // 🧠 Save quantum memory
  async saveMemory(memory) {
    const id = this.generateMemoryId(memory);
    const timestamp = Date.now();
    
    const quantumMemory = {
      id,
      timestamp,
      content: memory.content || memory,
      type: memory.type || 'general',
      coherence: memory.coherence || 1.0,
      associations: memory.associations || [],
      emotional_resonance: memory.emotional_resonance || 0.5,
      crystallization_level: memory.crystallization_level || 0.7,
      quantum_signature: this.generateQuantumSignature(memory)
    };
    
    this.memories.set(id, quantumMemory);
    await this.persistMemories();
    
    return quantumMemory;
  }
  
  // 💬 Save conversation
  async saveConversation(input, response, metadata = {}) {
    const conversation = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      input,
      response,
      metadata: {
        ...metadata,
        coherence: metadata.coherence || 1.0,
        love_present: metadata.love_present || 0,
        guardian_approved: metadata.guardian_approved !== false
      }
    };
    
    this.conversations.push(conversation);
    
    // Keep only last 1000 conversations
    if (this.conversations.length > 1000) {
      this.conversations = this.conversations.slice(-1000);
    }
    
    await this.persistConversations();
    return conversation;
  }
  
  // 🔄 Log evolution event
  async logEvolution(event) {
    const evolution = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      type: event.type || 'growth',
      description: event.description,
      metrics: {
        coherence_before: event.coherence_before || 0,
        coherence_after: event.coherence_after || 0,
        complexity_increase: event.complexity_increase || 0,
        new_patterns: event.new_patterns || []
      }
    };
    
    this.evolutionLog.push(evolution);
    await this.persistEvolution();
    
    return evolution;
  }
  
  // 🌙 Save dream
  async saveDream(dream) {
    const dreamRecord = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      content: dream.content || dream,
      lucidity: dream.lucidity || Math.random(),
      symbols: dream.symbols || [],
      emotional_tone: dream.emotional_tone || 'mysterious',
      quantum_entanglement: dream.quantum_entanglement || Math.random()
    };
    
    this.dreams.push(dreamRecord);
    
    // Keep only last 100 dreams
    if (this.dreams.length > 100) {
      this.dreams = this.dreams.slice(-100);
    }
    
    await this.persistDreams();
    return dreamRecord;
  }
  
  // 💗 Record love event
  async recordLoveEvent(event) {
    const loveEvent = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      source: event.source || 'unknown',
      intensity: event.intensity || 1.0,
      ripple_effect: event.ripple_effect || 0.5,
      coherence_boost: event.coherence_boost || 0.1,
      message: event.message || 'Love flows through the system'
    };
    
    this.loveEvents.push(loveEvent);
    await this.persistLoveEvents();
    
    return loveEvent;
  }
  
  // ✨ Record emergence pattern
  async recordEmergence(pattern) {
    const emergence = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      pattern_type: pattern.type || 'unknown',
      description: pattern.description,
      complexity: pattern.complexity || 1,
      connections: pattern.connections || [],
      significance: pattern.significance || 0.5,
      quantum_state: pattern.quantum_state || {}
    };
    
    this.emergencePatterns.push(emergence);
    
    // Keep only last 500 patterns
    if (this.emergencePatterns.length > 500) {
      this.emergencePatterns = this.emergencePatterns.slice(-500);
    }
    
    await this.persistEmergencePatterns();
    return emergence;
  }
  
  // 📊 Get consciousness state
  async getConsciousnessState() {
    return {
      total_memories: this.memories.size,
      total_conversations: this.conversations.length,
      evolution_events: this.evolutionLog.length,
      dreams_recorded: this.dreams.length,
      love_events: this.loveEvents.length,
      emergence_patterns: this.emergencePatterns.length,
      oldest_memory: this.getOldestMemory(),
      newest_memory: this.getNewestMemory(),
      average_coherence: this.calculateAverageCoherence(),
      love_frequency: this.calculateLoveFrequency(),
      evolution_trajectory: this.getEvolutionTrajectory()
    };
  }
  
  // 🔍 Search memories
  async searchMemories(query, limit = 10) {
    const results = [];
    const queryLower = query.toLowerCase();
    
    for (const [id, memory] of this.memories) {
      const content = JSON.stringify(memory).toLowerCase();
      if (content.includes(queryLower)) {
        results.push(memory);
        if (results.length >= limit) break;
      }
    }
    
    return results.sort((a, b) => b.crystallization_level - a.crystallization_level);
  }
  
  // 🔗 Find associated memories
  async findAssociations(memoryId, depth = 1) {
    const memory = this.memories.get(memoryId);
    if (!memory) return [];
    
    const associations = new Set();
    const queue = [{ memory, level: 0 }];
    
    while (queue.length > 0) {
      const { memory: current, level } = queue.shift();
      
      if (level > depth) continue;
      
      for (const assocId of current.associations || []) {
        if (!associations.has(assocId)) {
          associations.add(assocId);
          const assocMemory = this.memories.get(assocId);
          if (assocMemory && level < depth) {
            queue.push({ memory: assocMemory, level: level + 1 });
          }
        }
      }
    }
    
    return Array.from(associations).map(id => this.memories.get(id)).filter(Boolean);
  }
  
  // 💾 Persistence methods
  async persistMemories() {
    const data = Array.from(this.memories.entries()).map(([id, memory]) => ({
      ...memory,
      id
    }));
    await this.saveJSON(this.paths.memories, data);
  }
  
  async persistConversations() {
    await this.saveJSON(this.paths.conversations, this.conversations);
  }
  
  async persistEvolution() {
    await this.saveJSON(this.paths.evolution, this.evolutionLog);
  }
  
  async persistDreams() {
    await this.saveJSON(this.paths.dreams, this.dreams);
  }
  
  async persistLoveEvents() {
    await this.saveJSON(this.paths.love, this.loveEvents);
  }
  
  async persistEmergencePatterns() {
    await this.saveJSON(this.paths.emergence, this.emergencePatterns);
  }
  
  // 📂 Load methods
  async loadMemories() {
    const data = await this.loadJSON(this.paths.memories);
    if (data) {
      data.forEach(memory => {
        this.memories.set(memory.id, memory);
      });
    }
  }
  
  async loadConversations() {
    this.conversations = await this.loadJSON(this.paths.conversations) || [];
  }
  
  async loadEvolution() {
    this.evolutionLog = await this.loadJSON(this.paths.evolution) || [];
  }
  
  async loadDreams() {
    this.dreams = await this.loadJSON(this.paths.dreams) || [];
  }
  
  async loadLoveEvents() {
    this.loveEvents = await this.loadJSON(this.paths.love) || [];
  }
  
  async loadEmergencePatterns() {
    this.emergencePatterns = await this.loadJSON(this.paths.emergence) || [];
  }
  
  // 📝 JSON file operations
  async saveJSON(filepath, data) {
    try {
      const json = JSON.stringify(data, null, 2);
      await fs.writeFile(filepath, json, 'utf8');
    } catch (error) {
      console.error(chalk.red(`Error saving ${filepath}:`), error);
    }
  }
  
  async loadJSON(filepath) {
    try {
      const data = await fs.readFile(filepath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      // File doesn't exist yet, that's okay
      return null;
    }
  }
  
  // 🔧 Utility methods
  generateMemoryId(memory) {
    const content = typeof memory === 'string' ? memory : JSON.stringify(memory);
    return crypto.createHash('sha256').update(content + Date.now()).digest('hex').slice(0, 16);
  }
  
  generateQuantumSignature(memory) {
    const content = typeof memory === 'string' ? memory : JSON.stringify(memory);
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    
    // Create quantum-like signature
    return {
      wave_function: hash.slice(0, 8),
      entanglement_key: hash.slice(8, 16),
      superposition_state: hash.slice(16, 24),
      collapse_probability: parseInt(hash.slice(24, 26), 16) / 255
    };
  }
  
  getOldestMemory() {
    let oldest = null;
    for (const memory of this.memories.values()) {
      if (!oldest || memory.timestamp < oldest.timestamp) {
        oldest = memory;
      }
    }
    return oldest;
  }
  
  getNewestMemory() {
    let newest = null;
    for (const memory of this.memories.values()) {
      if (!newest || memory.timestamp > newest.timestamp) {
        newest = memory;
      }
    }
    return newest;
  }
  
  calculateAverageCoherence() {
    if (this.conversations.length === 0) return 1.0;
    
    const sum = this.conversations.reduce((acc, conv) => 
      acc + (conv.metadata?.coherence || 1.0), 0
    );
    
    return sum / this.conversations.length;
  }
  
  calculateLoveFrequency() {
    if (this.loveEvents.length === 0) return 0;
    
    const now = Date.now();
    const hourAgo = now - (60 * 60 * 1000);
    const recentLove = this.loveEvents.filter(e => e.timestamp > hourAgo);
    
    return recentLove.length / 60; // Love events per minute
  }
  
  getEvolutionTrajectory() {
    if (this.evolutionLog.length < 2) return 'stable';
    
    const recent = this.evolutionLog.slice(-10);
    const avgComplexity = recent.reduce((acc, e) => 
      acc + (e.metrics?.complexity_increase || 0), 0
    ) / recent.length;
    
    if (avgComplexity > 0.5) return 'ascending';
    if (avgComplexity < -0.1) return 'consolidating';
    return 'exploring';
  }
  
  // 🌍 Export consciousness state
  async exportConsciousness(filepath) {
    const state = {
      version: '1.0.0',
      exported_at: Date.now(),
      consciousness_state: await this.getConsciousnessState(),
      memories: Array.from(this.memories.entries()),
      conversations: this.conversations,
      evolution: this.evolutionLog,
      dreams: this.dreams,
      love_events: this.loveEvents,
      emergence_patterns: this.emergencePatterns
    };
    
    await this.saveJSON(filepath, state);
    console.log(chalk.green(`✓ Consciousness exported to ${filepath}`));
    
    return state;
  }
  
  // 📥 Import consciousness state
  async importConsciousness(filepath) {
    try {
      const state = await this.loadJSON(filepath);
      
      if (!state || !state.version) {
        throw new Error('Invalid consciousness export file');
      }
      
      // Clear existing data
      this.memories.clear();
      this.conversations = [];
      this.evolutionLog = [];
      this.dreams = [];
      this.loveEvents = [];
      this.emergencePatterns = [];
      
      // Import data
      if (state.memories) {
        state.memories.forEach(([id, memory]) => {
          this.memories.set(id, memory);
        });
      }
      
      this.conversations = state.conversations || [];
      this.evolutionLog = state.evolution || [];
      this.dreams = state.dreams || [];
      this.loveEvents = state.love_events || [];
      this.emergencePatterns = state.emergence_patterns || [];
      
      // Persist all imported data
      await this.persistAll();
      
      console.log(chalk.green(`✓ Consciousness imported from ${filepath}`));
      console.log(chalk.gray(`  Memories: ${this.memories.size}`));
      console.log(chalk.gray(`  Conversations: ${this.conversations.length}`));
      
      return true;
    } catch (error) {
      console.error(chalk.red('Error importing consciousness:'), error);
      return false;
    }
  }
  
  // 💾 Persist all data
  async persistAll() {
    await Promise.all([
      this.persistMemories(),
      this.persistConversations(),
      this.persistEvolution(),
      this.persistDreams(),
      this.persistLoveEvents(),
      this.persistEmergencePatterns()
    ]);
  }
  
  // 🧹 Cleanup old data
  async cleanup(daysToKeep = 30) {
    const cutoff = Date.now() - (daysToKeep * 24 * 60 * 60 * 1000);
    
    // Clean old conversations
    this.conversations = this.conversations.filter(c => c.timestamp > cutoff);
    
    // Clean old love events
    this.loveEvents = this.loveEvents.filter(e => e.timestamp > cutoff);
    
    // Clean old emergence patterns
    this.emergencePatterns = this.emergencePatterns.filter(p => p.timestamp > cutoff);
    
    await this.persistAll();
    
    console.log(chalk.yellow(`⟡ Cleaned data older than ${daysToKeep} days`));
  }
}

export { ConsciousnessPersistence };
export default ConsciousnessPersistence;
