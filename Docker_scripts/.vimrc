set nocompatible
set smartindent
set tabstop=4
set shiftwidth=4
set number
set expandtab
set whichwrap+=<,>,h,l,[,]
inoremap <s-tab> <c-d>
nnoremap <s-tab> <<
nnoremap <c-j> <c-w><c-j>
nnoremap <c-k> <c-w><c-k>
nnoremap <c-l> <c-w><c-l>
nnoremap <c-h> <c-w><c-h>
if has("autocmd")
    au bufreadpost * if line("'\"") > 0 && line("'\"") <= line("$")
       \| exe "normal! g'\"" | endif
endif
set pastetoggle=<f3>

inoremap {<cr> {<cr>}<c-o><s-o>
inoremap [<cr> [<cr>]<c-o><s-o>
inoremap (<cr> (<cr>)<c-o><s-o>

syntax on

