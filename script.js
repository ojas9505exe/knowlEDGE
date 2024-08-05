// Get the login and signup button elements
const loginButton = document.querySelector('.login');
const signupButton = document.querySelector('.signup');
const slider = document.querySelector('.slider');
const formSection = document.querySelector('.form-section');

// Event listeners for login and signup buttons
loginButton.addEventListener('click', function() {
    // Ensure login is handled here
    handleLogin();
});

signupButton.addEventListener('click', function() {
    // Redirect to signup page or handle signup here
    // Example redirect:
    // window.location.href = 'signup.html';
});

// Function to handle login
function handleLogin() {
    // Set the 'loggedIn' flag in localStorage
    localStorage.setItem('loggedIn', 'true');
    // Redirect to 1.html after setting the flag
    window.location.href = '1.html';
}

// Switch between login and signup forms
signupButton.addEventListener('click', () => {
    slider.classList.add('moveslider');
    formSection.classList.add('form-section-move');
});

loginButton.addEventListener('click', () => {
    slider.classList.remove('moveslider');
    formSection.classList.remove('form-section-move');
});
