# Cora: Heart Centered AI Assistant 🤖 + 💙

![Build Status](https://github.com/TechNickAI/cora/actions/workflows/build.yml/badge.svg)
![Lint Status](https://github.com/TechNickAI/cora/actions/workflows/linter.yml/badge.svg)

## Overview

Cora is a heart-centered AI assistant designed to revolutionize the way we interact with technology and enhance human potential. Named after the Latin word for "heart," Cora embodies our commitment to creating AI that is not just intelligent, but also empathetic, ethical, and aligned with human values.

Our vision is to develop an AI assistant that serves as a synergist between human creativity and technological capability. Cora is not just a tool, but a partner in your personal and professional growth, aimed at amplifying your effectiveness and helping you achieve a 10x improvement in various aspects of your life.

## Ethos and Goals

At the core of Cora's development is the philosophy of Heart Centered AI, which seeks to create a harmonious intersection between humanity and technology. Our goals include:

1. Empowerment: Enhance human capabilities, allowing users to focus on high-level strategy and creativity.
2. Ethical AI: Ensure that AI development remains grounded in empathy and multi-variate values.
3. Personalization: Provide deeply tailored assistance by leveraging personal data responsibly.
4. Efficiency: Dramatically improve productivity by handling a wide range of tasks with varying complexity.
5. Innovation: Push the boundaries of what's possible with AI, particularly in entrepreneurial and creative domains.
6. Holistic Growth: Support users in achieving balance across all aspects of life, not just career or productivity.

## Features and Roadmap

### Prototype Release (cli)

- [x] Basic cli interface with text input and output via rich
- [ ] Real time web searching (Exa, Tivaly, Perplexity)
- [x] Simple agent with tools
- [x] Pre-processing user query for enhanced prompt engineering
- [ ] Integration with Zep for chat history
- [x] Integration with multiple language models (Claude, GPT-4o, Gemini)
- [ ] Intelligent model selection based on query type

### Web Frontend Release: Foundation (on par with ChatGPT)

- [ ] React frontend with components from langui.dev
- [ ] Login with Google
- [ ] Basic personal data integration (location, preferences)
- [ ] Voice input

### Beta Release

- [ ] Customizable AI agent sophistication levels (dynamically created agents)
- [ ] Multi-agent conversations with multiple perspectives

### Version 1.0 (release to friends)

- [ ] Integration with Google services (Calendar, Contacts, Photos)
- [ ] Secure access to advanced tools (e.g., Python shell for trusted users)
- [ ] Full integration with social media platforms and task management tools

### Someday

- [ ] Comprehensive personal data analysis and insights (AI Chief of Life Officer)
- [ ] Integration with signed in browser profile (ala Multi-On) for logged in web tasks
- [ ] Auto-journaling and life-logging capabilities
- [ ] AI-driven project management and execution of complex tasks
- [ ] Collaborative AI sessions for multiple users

## Technology Stack

- Frontend React, langui.dev
- XXX for websocket communication
- AI Orchestration: Langchain, Langgraph
- Web searching via Exa, Tivaly, Perplexity
- Language Models: Claude, GPT-4, Gemini, and specialized models
- API Integrations: Google Services, Instagram, Facebook, Asana
- Memory via Zep
- pytest

## Installation and Setup

TODO

## Privacy and Security

Cora is designed with the highest standards of data privacy and security. While it leverages personal data for enhanced functionality, all information is processed securely, and users have full control over their data. This project is currently intended for use in a trusted environment and is not for public deployment.

## Acknowledgments

- Inspired by the concept of Heart Centered AI and the vision of creating technology that truly serves humanity
- Built with gratitude for the open-source AI community and pioneers in ethical AI development

## Coding Principles

Borrowed from the [zen of python](http://c2.com/cgi/wiki?PythonPhilosophy), with a couple of changes.

```text
1. **Readability is the number 1 code quality metric**.
2. Beautiful is better than ugly.
3. Explicit is better than implicit.
4. Simple is better than complex.
5. Complex is better than complicated.
6. Flat is better than nested.
7. Sparse is better than dense.
8. Special cases aren't special enough to break the rules.
    * Although practicality beats purity.
9. Errors should never pass silently.
    * Unless explicitly silenced.
10. In the face of ambiguity, refuse the temptation to guess.
11. There should be one -- and preferably only one -- obvious way to do it.
12. Now is better than never.
```
