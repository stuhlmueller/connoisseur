---
layout: default
---

{% assign sorted_pages = site.pages | sort:"name" %}

{% for p in sorted_pages %}
    {% if p.hidden %}
    {% else %}
        {% if p.layout == 'model' %}
1. <a class="model-link" href="{{ p.url | prepend: site.baseurl }}">{{ p.title }}</a> {% if p.author %}<span class="author">({{ p.author }})</span>{% endif %}
        {% endif %}
    {% endif %}
{% endfor %}