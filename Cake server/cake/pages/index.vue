<template>
    <div class="loader-container">
      <div class="loader"></div>
      <h1>Loading...</h1>
    </div>
  </template>
  
  <script>
  export default {
    mounted() {
      setTimeout(() => {
        this.$router.push('/menu');
      }, 3000); // Wait for 3 seconds before redirecting
    }
  }
  </script>
  
  <style scoped>
  .loader-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
  }
  
  .loader {
    border: 6px solid #f3f3f3; /* Light grey */
    border-top: 6px solid #3498db; /* Blue */
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 2s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  h1 {
    margin-top: 20px;
  }
  </style>
  