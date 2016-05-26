"use strict";


// Github links

var github_repository = "https://github.com/stuhlmueller/play/";

function markdown_url(page_url) {
  return page_url.slice(0, -4) + "md";
}

function github_edit_url(page_url) {
  return github_repository + "edit/gh-pages" + markdown_url(page_url);
}

function github_page_url(page_url) {
  if ((page_url == "/index.html") || (page_url == "/")) {
    return github_repository;
  } else {
    return github_repository + "blob/gh-pages" + markdown_url(page_url);
  };
}


// WebPPL editor

$(function(){
  var preEls = Array.prototype.slice.call(document.querySelectorAll("pre"));
  preEls.map(function(el) { wpEditor.setup(el, {language: 'webppl'}); });          
});


// Add code box link

$(function(){
  $('#add-code-box').click(function(e){
    e.preventDefault();
    var newElement = $('<pre></pre>').appendTo('div.page-content')[0];
    wpEditor.setup(newElement, {language: 'webppl'});
    return false;
  });
});
