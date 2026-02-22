console.log('Welcome to Tetris! Ready for Canvas or React implementation.'); 

// 俄罗斯方块核心参数
const COLS = 10;
const ROWS = 20;
const BLOCK_SIZE = 32;
const COLORS = [
  null,
  '#FF0D72', // I
  '#0DC2FF', // J
  '#0DFF72', // L
  '#F538FF', // O
  '#FF8E0D', // S
  '#FFE138', // T
  '#3877FF'  // Z
];

// 俄罗斯方块形状
const SHAPES = [
  [],
  [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ], // I
  [
    [2, 0, 0],
    [2, 2, 2],
    [0, 0, 0]
  ], // J
  [
    [0, 0, 3],
    [3, 3, 3],
    [0, 0, 0]
  ], // L
  [
    [4, 4],
    [4, 4]
  ], // O
  [
    [0, 5, 5],
    [5, 5, 0],
    [0, 0, 0]
  ], // S
  [
    [0, 6, 0],
    [6, 6, 6],
    [0, 0, 0]
  ], // T
  [
    [7, 7, 0],
    [0, 7, 7],
    [0, 0, 0]
  ]  // Z
];

// 获取Canvas和上下文
const canvas = document.getElementById('tetris-canvas');
const ctx = canvas.getContext('2d');
canvas.width = COLS * BLOCK_SIZE;
canvas.height = ROWS * BLOCK_SIZE;

// 游戏状态
let board = createMatrix(COLS, ROWS);
let score = 0;
let gameOver = false;

// 当前方块
let piece = null;

// 加载背景图片
const bgImg = new Image();
bgImg.src = 'background.jpg';
let bgLoaded = false;
bgImg.onload = () => {
  bgLoaded = true;
};

function createMatrix(w, h) {
  const matrix = [];
  for (let y = 0; y < h; ++y) {
    matrix.push(new Array(w).fill(0));
  }
  return matrix;
}

function drawMatrix(matrix, offset) {
  matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (value) {
        ctx.fillStyle = COLORS[value];
        ctx.fillRect(
          (x + offset.x) * BLOCK_SIZE,
          (y + offset.y) * BLOCK_SIZE,
          BLOCK_SIZE - 1,
          BLOCK_SIZE - 1
        );
      }
    });
  });
}

function merge(board, piece) {
  piece.matrix.forEach((row, y) => {
    row.forEach((value, x) => {
      if (value) {
        board[y + piece.pos.y][x + piece.pos.x] = value;
      }
    });
  });
}

function collide(board, piece) {
  const m = piece.matrix;
  const o = piece.pos;
  for (let y = 0; y < m.length; ++y) {
    for (let x = 0; x < m[y].length; ++x) {
      if (
        m[y][x] &&
        (board[y + o.y] && board[y + o.y][x + o.x]) !== 0
      ) {
        return true;
      }
    }
  }
  return false;
}

function rotate(matrix) {
  // 先转置再反转
  for (let y = 0; y < matrix.length; ++y) {
    for (let x = 0; x < y; ++x) {
      [matrix[x][y], matrix[y][x]] = [matrix[y][x], matrix[x][y]];
    }
  }
  matrix.forEach(row => row.reverse());
}

function playerDrop() {
  piece.pos.y++;
  if (collide(board, piece)) {
    piece.pos.y--;
    merge(board, piece);
    resetPiece();
    sweep();
    if (collide(board, piece)) {
      gameOver = true;
    }
  }
  dropCounter = 0;
}

function playerMove(dir) {
  piece.pos.x += dir;
  if (collide(board, piece)) {
    piece.pos.x -= dir;
  }
}

function playerRotate() {
  const pos = piece.pos.x;
  let offset = 1;
  rotate(piece.matrix);
  while (collide(board, piece)) {
    piece.pos.x += offset;
    offset = -(offset + (offset > 0 ? 1 : -1));
    if (offset > piece.matrix[0].length) {
      rotate(piece.matrix);
      rotate(piece.matrix);
      rotate(piece.matrix);
      piece.pos.x = pos;
      return;
    }
  }
}

function sweep() {
  let rowCount = 1;
  outer: for (let y = board.length - 1; y >= 0; --y) {
    for (let x = 0; x < board[y].length; ++x) {
      if (board[y][x] === 0) {
        continue outer;
      }
    }
    const row = board.splice(y, 1)[0].fill(0);
    board.unshift(row);
    score += rowCount * 10;
    rowCount *= 2;
    y++;
  }
}

function resetPiece() {
  const typeId = (Math.random() * (SHAPES.length - 1) + 1) | 0;
  piece = {
    matrix: SHAPES[typeId].map(row => row.slice()),
    pos: { x: ((COLS / 2) | 0) - ((SHAPES[typeId][0].length / 2) | 0), y: 0 }
  };
}

function drawBoard() {
  if (bgLoaded) {
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  drawMatrix(board, { x: 0, y: 0 });
  if (piece) drawMatrix(piece.matrix, piece.pos);
}

function drawScore() {
  ctx.font = '20px Arial';
  ctx.fillStyle = '#fff';
  ctx.fillText('Score: ' + score, 10, 30);
}

let dropCounter = 0;
let dropInterval = 500;
let lastTime = 0;

function update(time = 0) {
  if (gameOver) {
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '40px Arial';
    ctx.fillStyle = '#fff';
    ctx.fillText('Game Over', 40, canvas.height / 2);
    ctx.font = '24px Arial';
    ctx.fillText('Score: ' + score, 80, canvas.height / 2 + 40);
    ctx.fillText('F5刷新重玩', 70, canvas.height / 2 + 80);
    return;
  }
  const deltaTime = time - lastTime;
  lastTime = time;
  dropCounter += deltaTime;
  if (dropCounter > dropInterval) {
    playerDrop();
  }
  drawBoard();
  drawScore();
  requestAnimationFrame(update);
}

// 键盘事件
window.addEventListener('keydown', e => {
  if (gameOver) return;
  if (e.key === 'ArrowLeft') {
    playerMove(-1);
  } else if (e.key === 'ArrowRight') {
    playerMove(1);
  } else if (e.key === 'ArrowDown') {
    playerDrop();
  } else if (e.key === 'ArrowUp' || e.key === ' ') {
    playerRotate();
  }
});

// 初始化
resetPiece();
update(); 