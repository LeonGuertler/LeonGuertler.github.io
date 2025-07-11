---
layout: page
title: "👋 Welcome"
---

<pre>
{% assign bar = "+------------------------------+------------+" %}
{{ bar }}
| Post                         | Date       |
{{ bar }}
{% for post in site.posts %}
{%  comment %} pad each title to 28 chars {% endcomment %}
{% assign padded = post.title | append: "                            " | slice: 0, 28 %}
| {{ padded }} | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
</pre>
