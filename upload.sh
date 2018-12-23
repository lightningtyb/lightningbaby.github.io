cd themes/next/
git add .
git commit -m"modified thems"
git push https://github.com/lightningbaby/lightningbaby.github.io.git themes
cd ../../
hexo g
hexo d
git add .
git commit -m"add a news post"
git push https://github.com/lightningbaby/lightningbaby.github.io.git blog
