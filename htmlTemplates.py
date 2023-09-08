css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.0rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 90%;
  padding: 0 1.0rem;
  color: #fff;
}
'''

html_code = '''
<style>
    .fixed-text-input {
      position: fixed;
      top: 10px; /* Adjust top position as needed */
      left: 10px; /* Adjust left position as needed */
      z-index: 1000; /* Ensure it's above other content */
}
<style>
<div class="fixed-text-input">
  <input type="text" placeholder="Enter text">
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://github.com/marpulli246/llm_pdf_chatbot/blob/main/chatbot.png?raw=true" alt="chatbot.png" class="inline" style="width: 50px; height: 50px; margin-right: 10px;">
    </div>    
    <div class="message">{{MSG}}</div> 
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''
