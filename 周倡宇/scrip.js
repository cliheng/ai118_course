/**
 * RESUME SCRIPT - MAIN FUNCTIONALITY
 * Features:
 * 1. Dynamic date updating
 * 2. Print functionality
 * 3. Responsive adjustments
 * 4. Smooth animations
 */

document.addEventListener('DOMContentLoaded', function() {
    // ===== DYNAMIC DATE =====
    const setUpdateDate = () => {
        const currentDate = new Date();
        const options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        document.getElementById('update-date').textContent = 
            currentDate.toLocaleDateString('zh-CN', options);
    };
    
    // ===== PRINT FUNCTIONALITY =====
    const initPrintButton = () => {
        const printBtn = document.createElement('button');
        printBtn.id = 'print-btn';
        printBtn.className = 'no-print';
        printBtn.innerHTML = '<i class="fas fa-print"></i> 打印简历';
        
        // Style the button
        Object.assign(printBtn.style, {
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            padding: '10px 15px',
            background: '#6e8efb',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            zIndex: '1000',
            boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
            transition: 'all 0.3s ease'
        });
        
        // Hover effects
        printBtn.addEventListener('mouseenter', () => {
            printBtn.style.background = '#5a7df4';
            printBtn.style.transform = 'translateY(-2px)';
        });
        
        printBtn.addEventListener('mouseleave', () => {
            printBtn.style.background = '#6e8efb';
            printBtn.style.transform = 'translateY(0)';
        });
        
        // Click handler
        printBtn.addEventListener('click', () => {
            window.print();
        });
        
        document.body.appendChild(printBtn);
    };
    
    // ===== SOCIAL LINKS HANDLER =====
    const initSocialLinks = () => {
        const socialLinks = document.querySelectorAll('[data-social]');
        
        socialLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const platform = link.getAttribute('data-social');
                const urlMap = {
                    'github': 'https://github.com/yourusername',
                    'linkedin': 'https://linkedin.com/in/yourprofile'
                };
                
                if (urlMap[platform]) {
                    window.open(urlMap[platform], '_blank');
                }
            });
        });
    };
    
    // ===== SKILL LEVEL BARS =====
    const initSkillBars = () => {
        const skills = {
            'Python': 90,
            'Java': 80,
            'SQL': 95,
            'English': 85,
            'Japanese': 70
        };
        
        Object.entries(skills).forEach(([skill, level]) => {
            const element = document.querySelector(`[data-skill="${skill}"]`);
            if (element) {
                element.style.width = `${level}%`;
                element.setAttribute('data-level', `${level}%`);
            }
        });
    };
    
    // ===== INITIALIZATION =====
    setUpdateDate();
    initPrintButton();
    initSocialLinks();
    initSkillBars();
    
    // ===== PERFORMANCE OPTIMIZATION =====
    requestAnimationFrame(() => {
        document.body.style.opacity = '1';
    });
});