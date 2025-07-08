// 导航栏滚动效果
window.addEventListener('scroll', function() {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// 移动端菜单切换
const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links');

menuToggle.addEventListener('click', function() {
    navLinks.classList.toggle('active');
});

// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        
        // 关闭移动端菜单
        if (navLinks.classList.contains('active')) {
            navLinks.classList.remove('active');
        }
        
        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        
        window.scrollTo({
            top: targetElement.offsetTop - 70,
            behavior: 'smooth'
        });
    });
});

// 动画效果 - 滚动时显示元素
function animateOnScroll() {
    const elements = document.querySelectorAll('.timeline-item, .skills-category, .project-card, .certificate-card');
    
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const screenPosition = window.innerHeight / 1.3;
        
        if (elementPosition < screenPosition) {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }
    });
}

// 设置初始动画状态
window.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('.timeline-item, .skills-category, .project-card, .certificate-card');
    
    elements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'all 0.6s ease';
    });
    
    // 触发一次检查，确保首屏外的元素初始状态正确
    animateOnScroll();
});

// 滚动时触发动画
window.addEventListener('scroll', animateOnScroll);

// 表单提交
const contactForm = document.querySelector('.contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 这里可以添加表单验证逻辑
        const formData = new FormData(this);
        
        // 模拟表单提交
        setTimeout(() => {
            alert('感谢您的留言！我会尽快回复您。');
            this.reset();
        }, 500);
    });
}