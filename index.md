<pre>
{% assign bar = "+------------------------------+------------+" %}
{{ bar }}
| Post                         | Date       |
{{ bar }}
{% for post in site.posts %}
{%  comment %} pad each title to 28 chars {% endcomment %}
{% assign padded = post.title | append: "                            " | slice: 0, 28 %}

| <a href="{{ post.url | relative_url }}" style="color:inherit;text-decoration:none;">{{ padded }}</a> | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
</pre>


<pre markdown="1">
{% assign bar = "+------------------------------+------------+" %}
{{ bar }}
| Post                         | Date       |
{{ bar }}
{% assign pad = "                              " %}  <!-- 30 spaces -->
{% for post in site.posts %}
| [{{ post.title | append: pad | slice: 0, 28 }}]({{ post.url | relative_url }}) | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
</pre>



<pre markdown="1">
{%- assign bar = "+------------------------------+------------+" -%}
{{ bar }}
| Post                         | Date       |
{{ bar }}
{%- assign pad = "                              " -%}  <!-- 30 spaces -->
{%- for post in site.posts -%}
| [{{ post.title | append: pad | slice: 0, 28 }}]({{ post.url | relative_url }}) | {{ post.date | date: "%Y-%m-%d" }} |
{%- endfor -%}
{{ bar }}
</pre>