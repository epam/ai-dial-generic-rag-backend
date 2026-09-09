alter table text_chunks
    add column if not exists metadata jsonb not null default '{}'::jsonb;

update text_chunks
set metadata=jsonb_build_object('page_number', page_number);

alter table text_chunks
    drop column page_number;

alter table image_chunks
    add column if not exists metadata jsonb not null default '{}'::jsonb;

update image_chunks
set metadata=jsonb_build_object('page_number', page_number);

alter table image_chunks
    drop column page_number;
