document.addEventListener('DOMContentLoaded',  function() {
    // 导航栏滚动效果 
    const navbar = document.getElementById('navbar'); 
    const backToTop = document.querySelector('.back-to-top'); 
    
    window.addEventListener('scroll',  function() {
        if (window.scrollY  > 100) {
            navbar.classList.add('scrolled'); 
            backToTop.classList.add('active'); 
        } else {
            navbar.classList.remove('scrolled'); 
            backToTop.classList.remove('active'); 
        }
    });
 
    // 移动端菜单切换 
    const hamburger = document.querySelector('.hamburger'); 
    const navLinks = document.querySelector('.nav-links'); 
    
    hamburger.addEventListener('click',  function() {
        this.classList.toggle('active'); 
        navLinks.classList.toggle('active'); 
    });
 
    // 平滑滚动 
    document.querySelectorAll('a[href^="#"]').forEach(anchor  => {
        anchor.addEventListener('click',  function(e) {
            e.preventDefault(); 
            
            const targetId = this.getAttribute('href'); 
            const targetElement = document.querySelector(targetId); 
            
            if (targetElement) {
                window.scrollTo({ 
                    top: targetElement.offsetTop  - 70,
                    behavior: 'smooth'
                });
                
                // 关闭移动菜单 
                hamburger.classList.remove('active'); 
                navLinks.classList.remove('active'); 
            }
        });
    });
 
    // 导航链接高亮 
    const sections = document.querySelectorAll('section'); 
    const navItems = document.querySelectorAll('.nav-link'); 
    
    window.addEventListener('scroll',  function() {
        let current = '';
        
        sections.forEach(section  => {
            const sectionTop = section.offsetTop; 
            const sectionHeight = section.clientHeight; 
            
            if (window.scrollY  >= sectionTop - 300) {
                current = section.getAttribute('id'); 
            }
        });
        
        navItems.forEach(item  => {
            item.classList.remove('active'); 
            if (item.getAttribute('href')  === `#${current}`) {
                item.classList.add('active'); 
            }
        });
    });
 
    // 技能条动画 
    const skillBars = document.querySelectorAll('.skill-progress'); 
    
    function animateSkillBars() {
        skillBars.forEach(bar  => {
            const width = bar.style.width; 
            bar.style.width  = '0';
            
            setTimeout(() => {
                bar.style.width  = width;
            }, 100);
        });
    }
    
    // 使用IntersectionObserver来触发动画 
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry  => {
            if (entry.isIntersecting)  {
                animateSkillBars();
                observer.unobserve(entry.target); 
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('#skills').forEach(section  => {
        observer.observe(section); 
    });
 
    // 联系表单提交 
    const contactForm = document.getElementById('contactForm'); 
    
    if (contactForm) {
        contactForm.addEventListener('submit',  function(e) {
            e.preventDefault(); 
            
            // 获取表单数据 
            const formData = new FormData(this);
            const data = Object.fromEntries(formData); 
            
            // 这里可以添加AJAX请求来发送表单数据 
            console.log(' 表单已提交:', data);
            
            // 显示成功消息 
            alert('感谢您的留言！我会尽快回复您。');
            this.reset(); 
        });
    }
 
    // 页面加载动画 
    const animatedElements = document.querySelectorAll('.fade-in-up'); 
    
    animatedElements.forEach((element,  index) => {
        element.style.opacity  = '0';
        
        setTimeout(() => {
            element.classList.add('fade-in-up'); 
        }, index * 200);
    });
});